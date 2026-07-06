#pragma once

// Qwen3-0.6B 模型的数据结构：超参数 + 各层权重张量指针。
//
// v0.1 阶段的简化：
//   - 只支持从 GGUF 加载到 CPU backend。
//   - 权重张量的元数据（形状/名字）由 gguf_init_from_file() 自动建立在 ctx_data 里，
//     我们只需要按名字把指针取出来（见 model.cpp 的 get_tensor 辅助函数）。

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

// 从 GGUF 元数据读取的 Qwen3 超参数。
struct qwen3_hparams {
    int32_t n_vocab        = 0; // 词表大小（从 token_embd 的形状读取，比元数据更可靠）
    int32_t n_embd         = 0; // hidden_size
    int32_t n_layer        = 0; // Transformer block 数
    int32_t n_head         = 0; // Query 注意力头数
    int32_t n_head_kv      = 0; // KV 头数（GQA，<= n_head）
    int32_t n_embd_head    = 0; // 每个头的维度 head_dim（注意：!= n_embd/n_head）
    int32_t n_ff           = 0; // FFN 中间维度
    int32_t n_ctx_train    = 0; // 训练时的上下文长度上限
    float   rms_norm_eps   = 1e-6f;
    float   rope_freq_base = 1000000.0f;

    // 特殊 token id，v0.1 用来判断生成是否应该停止（分词本身仍由 Python 预处理）。
    int32_t eos_token_id = -1;
    int32_t bos_token_id = -1;

    // Q/K/V 投影展平后的维度：head_dim * n_head(_kv)，注意不等于 n_embd。
    int32_t n_embd_head_all() const { return n_embd_head * n_head; }
    int32_t n_embd_kv_all()   const { return n_embd_head * n_head_kv; }
};

// 单个 Transformer block 的权重。张量名与形状对照见 doc/design.md §7。
struct qwen3_layer {
    struct ggml_tensor * attn_norm   = nullptr; // [n_embd]                       输入 RMSNorm
    struct ggml_tensor * wq          = nullptr; // [n_embd, head_dim*n_head]      Q 投影
    struct ggml_tensor * wk          = nullptr; // [n_embd, head_dim*n_head_kv]   K 投影
    struct ggml_tensor * wv          = nullptr; // [n_embd, head_dim*n_head_kv]   V 投影
    struct ggml_tensor * wo          = nullptr; // [head_dim*n_head, n_embd]      输出投影
    struct ggml_tensor * attn_q_norm = nullptr; // [head_dim]                     QK-Norm（Qwen3 专有）
    struct ggml_tensor * attn_k_norm = nullptr; // [head_dim]                     QK-Norm（Qwen3 专有）
    struct ggml_tensor * ffn_norm    = nullptr; // [n_embd]                       FFN 前 RMSNorm
    struct ggml_tensor * ffn_gate    = nullptr; // [n_embd, n_ff]                 SwiGLU gate
    struct ggml_tensor * ffn_up      = nullptr; // [n_embd, n_ff]                 SwiGLU up
    struct ggml_tensor * ffn_down    = nullptr; // [n_ff, n_embd]                 SwiGLU down
};

struct qwen3_model {
    qwen3_hparams hparams;

    struct ggml_tensor * tok_embd    = nullptr; // [n_embd, n_vocab] token embedding
    struct ggml_tensor * output_norm = nullptr; // [n_embd]          末端 RMSNorm
    struct ggml_tensor * output      = nullptr; // [n_embd, n_vocab] lm_head，若 GGUF 缺失则等于 tok_embd（tied embedding）

    std::vector<qwen3_layer> layers;

    // ggml 资源：ctx_data 持有所有权重张量的元数据，buffer 是实际存放权重数据的 backend 内存块。
    struct ggml_context   * ctx_data = nullptr;
    ggml_backend_t          backend  = nullptr;
    ggml_backend_buffer_t   buffer   = nullptr;

    qwen3_model() = default;
    qwen3_model(const qwen3_model &) = delete;
    qwen3_model & operator=(const qwen3_model &) = delete;

    ~qwen3_model();
};

// 从 GGUF 文件加载 Qwen3 模型权重到 CPU backend。失败时返回 false 并打印原因到 stderr。
bool qwen3_model_load(const std::string & fname, qwen3_model & model);
