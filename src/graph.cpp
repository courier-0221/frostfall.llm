#include "graph.h"

#include <cmath>

// 单层前向的伪代码见 doc/design.md §2.3；GQA/RoPE/QK-Norm/因果 mask 的张量摆放细节见 §6。
struct ggml_cgraph * qwen3_build_graph(
        struct ggml_context * ctx,
        const qwen3_model    & model,
        int32_t                n_tokens,
        int                    max_nodes) {
    const qwen3_hparams & hp = model.hparams;

    const int64_t n_embd      = hp.n_embd;
    const int64_t n_head      = hp.n_head;
    const int64_t n_head_kv   = hp.n_head_kv;
    const int64_t n_embd_head = hp.n_embd_head;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // ---- 输入张量 ----
    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(tokens, QWEN3_TENSOR_NAME_TOKENS);
    ggml_set_input(tokens);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, QWEN3_TENSOR_NAME_POS);
    ggml_set_input(positions);

    // 因果 mask：ne0 = kv 位置(k)，ne1 = query 位置(q)；k > q（未来）的位置填 -inf。
    struct ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(kq_mask, QWEN3_TENSOR_NAME_MASK);
    ggml_set_input(kq_mask);

    // token embedding：[n_embd, n_tokens]
    struct ggml_tensor * x = ggml_get_rows(ctx, model.tok_embd, tokens);

    const float kq_scale = 1.0f / sqrtf((float) n_embd_head);

    for (int32_t il = 0; il < hp.n_layer; ++il) {
        const qwen3_layer & layer = model.layers[il];
        struct ggml_tensor * cur;

        // ==================== 自注意力 ====================
        struct ggml_tensor * inp_attn = x;

        // 输入 RMSNorm
        cur = ggml_rms_norm(ctx, x, hp.rms_norm_eps);
        cur = ggml_mul(ctx, cur, layer.attn_norm);

        // QKV 投影：注意展平维度是 head_dim*n_head(_kv)，不是 n_embd
        struct ggml_tensor * qcur = ggml_mul_mat(ctx, layer.wq, cur); // [head_dim*n_head,    n_tokens]
        struct ggml_tensor * kcur = ggml_mul_mat(ctx, layer.wk, cur); // [head_dim*n_head_kv, n_tokens]
        struct ggml_tensor * vcur = ggml_mul_mat(ctx, layer.wv, cur); // [head_dim*n_head_kv, n_tokens]

        qcur = ggml_reshape_3d(ctx, qcur, n_embd_head, n_head,    n_tokens);
        kcur = ggml_reshape_3d(ctx, kcur, n_embd_head, n_head_kv, n_tokens);
        vcur = ggml_reshape_3d(ctx, vcur, n_embd_head, n_head_kv, n_tokens);

        // QK-Norm（Qwen3 专有）：对每个头的 head_dim 向量单独做 RMSNorm，必须在 RoPE 之前。
        qcur = ggml_rms_norm(ctx, qcur, hp.rms_norm_eps);
        qcur = ggml_mul(ctx, qcur, layer.attn_q_norm);

        kcur = ggml_rms_norm(ctx, kcur, hp.rms_norm_eps);
        kcur = ggml_mul(ctx, kcur, layer.attn_k_norm);

        // RoPE（NEOX 变体，Qwen/LLaMA 系用这个，不能用 NORMAL，否则输出乱码）
        qcur = ggml_rope_ext(ctx, qcur, positions, nullptr, (int) n_embd_head, GGML_ROPE_TYPE_NEOX,
                              hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        kcur = ggml_rope_ext(ctx, kcur, positions, nullptr, (int) n_embd_head, GGML_ROPE_TYPE_NEOX,
                              hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // 把“头”维度换到第 3 维，方便 mul_mat 在头维上做 batched 矩阵乘：
        //   q: [head_dim, n_tokens, n_head]      k: [head_dim, n_tokens, n_head_kv]
        struct ggml_tensor * q = ggml_permute(ctx, qcur, 0, 2, 1, 3);
        struct ggml_tensor * k = ggml_permute(ctx, kcur, 0, 2, 1, 3);

        // kq = K^T Q：[n_tokens(kv), n_tokens(q), n_head]
        // n_head_kv < n_head 时，ggml_mul_mat 按 ne2 自动 broadcast（GQA：每组 Q 头共享一组 KV）。
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // V 转置成 [n_tokens(kv), head_dim, n_head_kv]，方便下一步 mul_mat
        struct ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, vcur, 1, 2, 0, 3));

        // kqv = V^T * softmax(kq)：[head_dim, n_tokens(q), n_head]
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);

        // 合并回 [head_dim*n_head, n_tokens]，供输出投影使用
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3); // -> [head_dim, n_head, n_tokens]
        cur = ggml_cont_2d(ctx, kqv, n_embd_head * n_head, n_tokens);

        // 输出投影 + 残差（wo 把 head_dim*n_head 映射回 n_embd）
        cur = ggml_mul_mat(ctx, layer.wo, cur);
        x = ggml_add(ctx, inp_attn, cur);

        // ==================== FFN（SwiGLU）====================
        struct ggml_tensor * inp_ffn = x;

        cur = ggml_rms_norm(ctx, x, hp.rms_norm_eps);
        cur = ggml_mul(ctx, cur, layer.ffn_norm);

        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur); // [n_ff, n_tokens]
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up,   cur); // [n_ff, n_tokens]
        gate = ggml_silu(ctx, gate);
        cur  = ggml_mul(ctx, gate, up);
        cur  = ggml_mul_mat(ctx, layer.ffn_down, cur); // [n_embd, n_tokens]

        x = ggml_add(ctx, inp_ffn, cur);
    }

    // ==================== 末端 RMSNorm + lm_head ====================
    x = ggml_rms_norm(ctx, x, hp.rms_norm_eps);
    x = ggml_mul(ctx, x, model.output_norm);

    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.output, x); // [n_vocab, n_tokens]
    ggml_set_name(logits, QWEN3_TENSOR_NAME_LOGITS);
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    return gf;
}
