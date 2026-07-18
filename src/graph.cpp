#include "graph.h"

#include <cmath>

// 单层前向的伪代码见 doc/design.md §2.3；GQA/RoPE/QK-Norm/因果 mask 的张量摆放细节见 §6。
// v0.2 相比 v0.1 的核心改动：K/V 不再每层重算整段，而是把本次 n_tokens 个 token 的 K/V
// 写入 kv cache 的 [n_past, n_past+n_tokens)，注意力从 cache 读取 [0, n_kv) 的全部历史。
struct ggml_cgraph * qwen3_build_graph(
        struct ggml_context * ctx,
        const qwen3_model    & model,
        const qwen3_kv_cache & kv,
        int32_t                n_tokens,
        int32_t                n_past,
        int                    max_nodes) {
    const qwen3_hparams & hp = model.hparams;

    const int64_t n_embd        = hp.n_embd;
    const int64_t n_head        = hp.n_head;
    const int64_t n_head_kv     = hp.n_head_kv;
    const int64_t n_embd_head   = hp.n_embd_head;
    const int64_t n_embd_kv_all = hp.n_embd_kv_all(); // head_dim * n_head_kv
    const int64_t n_kv          = n_past + n_tokens;  // 本次注意力可见的历史长度

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // ---- 输入张量 ----
    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(tokens, QWEN3_TENSOR_NAME_TOKENS);
    ggml_set_input(tokens);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, QWEN3_TENSOR_NAME_POS);
    ggml_set_input(positions);

    // 因果 mask：ne0 = kv 位置(0..n_kv-1)，ne1 = query 位置(本次的 n_tokens 个)；不可见处填 -inf。
    struct ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    ggml_set_name(kq_mask, QWEN3_TENSOR_NAME_MASK);
    ggml_set_input(kq_mask);

    // token embedding：[n_embd, n_tokens]
    struct ggml_tensor * x = ggml_get_rows(ctx, model.tok_embd, tokens);

    const float kq_scale = 1.0f / sqrtf((float) n_embd_head);
    const size_t kv_esz  = ggml_element_size(kv.k[0]); // cache 元素字节数（F16 = 2）

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

        // RoPE（NEOX 变体，Qwen/LLaMA 系用这个，不能用 NORMAL，否则输出乱码）。
        // 位置来自 positions（值 = n_past + i），保证 decode 阶段用的是绝对位置。
        qcur = ggml_rope_ext(ctx, qcur, positions, nullptr, (int) n_embd_head, GGML_ROPE_TYPE_NEOX,
                              hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        kcur = ggml_rope_ext(ctx, kcur, positions, nullptr, (int) n_embd_head, GGML_ROPE_TYPE_NEOX,
                              hp.n_ctx_train, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // ---- 把本次的 K/V 写入 cache 的 [n_past, n_past+n_tokens) ----
        // cache 是 1D，逻辑上每个 token 占 n_embd_kv_all 个元素；用 view_1d 定位到写入区间后 cpy 进去。
        struct ggml_tensor * k_dst = ggml_view_1d(ctx, kv.k[il],
                n_tokens * n_embd_kv_all, n_past * n_embd_kv_all * kv_esz);
        struct ggml_tensor * v_dst = ggml_view_1d(ctx, kv.v[il],
                n_tokens * n_embd_kv_all, n_past * n_embd_kv_all * kv_esz);
        // 立即 expand 这两个写 cache 的节点，确保它们排在后面“读 cache”的节点之前执行。
        ggml_build_forward_expand(gf, ggml_cpy(ctx, kcur, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, vcur, v_dst));

        // ---- 从 cache 读取 [0, n_kv) 的历史 K/V ----
        // K 看成 [head_dim, n_kv, n_head_kv]：内存里每个 kv 位置连续存 n_embd_kv_all 个值，
        //   nb1（相邻 kv 位置的跨度）= n_embd_kv_all 个元素，nb2（相邻头的跨度）= head_dim 个元素。
        struct ggml_tensor * K = ggml_view_3d(ctx, kv.k[il],
                n_embd_head, n_kv, n_head_kv,
                n_embd_kv_all * kv_esz,   // nb1
                n_embd_head   * kv_esz,   // nb2
                0);

        // Q 摆成 [head_dim, n_tokens, n_head]，方便在“头”维上做 batched 矩阵乘。
        struct ggml_tensor * q = ggml_permute(ctx, qcur, 0, 2, 1, 3);

        // kq = K^T Q：[n_kv, n_tokens, n_head]；n_head_kv < n_head 时 ggml_mul_mat 按 ne2 自动 broadcast（GQA）。
        struct ggml_tensor * kq = ggml_mul_mat(ctx, K, q);
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // V 从 cache 读成 [head_dim, n_kv, n_head_kv]，再转成 [n_kv, head_dim, n_head_kv] 供下一步 mul_mat。
        struct ggml_tensor * V = ggml_view_3d(ctx, kv.v[il],
                n_embd_head, n_kv, n_head_kv,
                n_embd_kv_all * kv_esz,
                n_embd_head   * kv_esz,
                0);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3)); // -> [n_kv, head_dim, n_head_kv]

        // kqv = V^T * softmax(kq)：[head_dim, n_tokens(q), n_head]
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);

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
