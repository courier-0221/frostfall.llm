#include "embed_graph.h"

#include <cmath>

// v0.1 单条前向：无 KV cache、无 lm_head、无 batch。
// 除"每层不写 KV cache"与"末尾不做 lm_head 投影"外，其余算子与 llm 版 graph.cpp 完全一致，
// 便于逐行对照理解。
struct ggml_cgraph * qwen3_embed_build_graph(
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
    ggml_set_name(tokens, QWEN3_EMBED_TENSOR_NAME_TOKENS);
    ggml_set_input(tokens);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, QWEN3_EMBED_TENSOR_NAME_POS);
    ggml_set_input(positions);

    // 因果 mask：ne0 = key 位置(0..n_tokens-1)，ne1 = query 位置(0..n_tokens-1)；未来位置填 -inf。
    // v0.1 没有 padding，n_kv = n_tokens，query 与 key 一一对齐。
    struct ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(kq_mask, QWEN3_EMBED_TENSOR_NAME_MASK);
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

        // QKV 投影：展平维度是 head_dim*n_head(_kv)，不是 n_embd
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

        // RoPE（NEOX 变体）。位置来自 positions（0..n_tokens-1）。
        qcur = ggml_rope_ext(ctx, qcur, positions, nullptr, (int) n_embd_head, GGML_ROPE_TYPE_NEOX,
                              hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        kcur = ggml_rope_ext(ctx, kcur, positions, nullptr, (int) n_embd_head, GGML_ROPE_TYPE_NEOX,
                              hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // ---- 注意力（无 KV cache）：K/V 直接用本层刚算出来的 kcur/vcur ----
        // K 已经是 [head_dim, n_head_kv, n_tokens]（reshape 后），转成 [head_dim, n_tokens, n_head_kv]
        // 供 mul_mat：把"kv 位置"放到 ne1、"kv 头"放到 ne2。
        struct ggml_tensor * K = ggml_permute(ctx, kcur, 0, 2, 1, 3); // [head_dim, n_tokens, n_head_kv]

        // Q 摆成 [head_dim, n_tokens, n_head]，"query 位置"放 ne1、"query 头"放 ne2。
        struct ggml_tensor * q = ggml_permute(ctx, qcur, 0, 2, 1, 3);

        // kq = K^T Q：[n_tokens(kv), n_tokens(q), n_head]；n_head_kv < n_head 时按 ne2 自动 broadcast（GQA）。
        struct ggml_tensor * kq = ggml_mul_mat(ctx, K, q);
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // V 摆成 [n_tokens, head_dim, n_head_kv]，供 kqv = V^T * softmax(kq)。
        // vcur 当前是 [head_dim, n_head_kv, n_tokens]，先 permute 成 [head_dim, n_tokens, n_head_kv]
        // 再 permute 成 [n_tokens, head_dim, n_head_kv]，需要 cont 让内存连续。
        struct ggml_tensor * V = ggml_permute(ctx, vcur, 1, 2, 0, 3); // [n_tokens, head_dim, n_head_kv]
        V = ggml_cont(ctx, V);

        // kqv = V^T * softmax(kq)：[head_dim, n_tokens(q), n_head]
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);

        // 合并回 [head_dim*n_head, n_tokens]，供输出投影使用
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3); // -> [head_dim, n_head, n_tokens]
        cur = ggml_cont_2d(ctx, kqv, n_embd_head * n_head, n_tokens);

        // 输出投影 + 残差
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

    // ==================== 末端 RMSNorm（无 lm_head）====================
    x = ggml_rms_norm(ctx, x, hp.rms_norm_eps);
    x = ggml_mul(ctx, x, model.output_norm);

    // 直接把 [n_embd, n_tokens] 作为图输出，交给 pooling 层处理。
    ggml_set_name(x, QWEN3_EMBED_TENSOR_NAME_HIDDEN);
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    return gf;
}
