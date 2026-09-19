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
    int32_t eos_token_id = -1; // <|endoftext|>（151643）
    int32_t eot_token_id = -1; // <|im_end|>（151645），chat 模式的实际结束符
    int32_t bos_token_id = -1;

    // Q/K/V 投影展平后的维度：head_dim * n_head(_kv)，注意不等于 n_embd。
    int32_t n_embd_head_all() const { return n_embd_head * n_head; }
    int32_t n_embd_kv_all()   const { return n_embd_head * n_head_kv; }
};

// ============================================================
// ASR（asr/v0.1）
//
// 架构 qwen3-asr：文本 Decoder 与 Qwen3 完全同构（复用上面的 hparams/layers），
// 额外携带音频塔权重。v0.1 只加载与校验音频塔张量（存在性 + 形状），
// 不参与计算图；音频编码在 asr/v0.2 实现。
// 张量名与形状契约见 tools/convert_asr_hf_to_gguf.py 与
// doc/asr/asr_v01_tensor_mapping.json。
// ============================================================

// ASR 音频塔超参数（来自 GGUF 元数据 asr.audio.* / asr.*）。
struct qwen3_asr_hparams {
    int32_t n_audio_layer = 0;  // encoder_layers（18）
    int32_t d_model       = 0;  // 音频 Transformer 隐藏宽度（896）
    int32_t n_head        = 0;  // encoder_attention_heads（14）
    int32_t n_ff          = 0;  // encoder_ffn_dim（3584）
    int32_t output_dim    = 0;  // 对齐文本 Decoder 的输出宽度（1024）
    int32_t num_mel_bins  = 0;  // 128
    int32_t n_window      = 0;  // CNN 分块 = 2*n_window 个 Mel 帧（50 -> 100）
    int32_t n_window_infer = 0; // 推理注意力窗口参数（800）
    int32_t conv_chunksize = 0; // 卷积分块上限（500）
    int32_t max_source_positions = 0; // 正弦位置表长度（1500）
    int32_t conv_channels = 0; // CNN 输出通道数（480），不在 config 中，加载时从 conv1.weight ne[3] 推导
    int32_t conv_kernel   = 3;
    int32_t conv_stride   = 2;
    int32_t conv_padding  = 1;
    float   layer_norm_eps = 1e-5f;

    // ASR 协议 token
    int32_t audio_start_token_id = -1; // <|audio_start|>（151669）
    int32_t audio_end_token_id   = -1; // <|audio_end|>（151670）
    int32_t audio_token_id       = -1; // <|audio_pad|>（151676），会被音频特征替换
    std::vector<int32_t> eos_token_ids; // 生成 EOS 集合（151645, 151643）
};

// 单个音频 Transformer block 的权重（attention 带 bias，LayerNorm 带 bias，普通两层 FFN）。
struct qwen3_asr_audio_layer {
    struct ggml_tensor * attn_q_w    = nullptr; // [896,896]
    struct ggml_tensor * attn_q_b    = nullptr; // [896]
    struct ggml_tensor * attn_k_w    = nullptr; // [896,896]
    struct ggml_tensor * attn_k_b    = nullptr;
    struct ggml_tensor * attn_v_w    = nullptr;
    struct ggml_tensor * attn_v_b    = nullptr;
    struct ggml_tensor * attn_out_w  = nullptr;
    struct ggml_tensor * attn_out_b  = nullptr;
    struct ggml_tensor * attn_norm_w = nullptr; // self_attn_layer_norm
    struct ggml_tensor * attn_norm_b = nullptr;
    struct ggml_tensor * fc1_w       = nullptr; // [3584,896]
    struct ggml_tensor * fc1_b       = nullptr; // [3584]
    struct ggml_tensor * fc2_w       = nullptr; // [896,3584]
    struct ggml_tensor * fc2_b       = nullptr; // [896]
    struct ggml_tensor * final_norm_w = nullptr; // final_layer_norm
    struct ggml_tensor * final_norm_b = nullptr;
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

    // ---- ASR（arch == "qwen3-asr" 时加载；v0.1 仅校验，不计算）----
    bool is_asr = false;
    qwen3_asr_hparams asr_hparams;

    // 音频塔全局张量（命名空间 asr.audio.*）
    struct ggml_tensor * a_conv1_w = nullptr; // [3,3,1,480]   ggml_conv_2d/im2col filter 布局
    struct ggml_tensor * a_conv1_b = nullptr; // [480]
    struct ggml_tensor * a_conv2_w = nullptr; // [3,3,480,480]
    struct ggml_tensor * a_conv2_b = nullptr;
    struct ggml_tensor * a_conv3_w = nullptr; // [3,3,480,480]
    struct ggml_tensor * a_conv3_b = nullptr;
    struct ggml_tensor * a_conv_out_w = nullptr; // [7680,896]，无 bias
    struct ggml_tensor * a_ln_post_w = nullptr;  // [896]
    struct ggml_tensor * a_ln_post_b = nullptr;
    struct ggml_tensor * a_proj1_w = nullptr; // [896,896]
    struct ggml_tensor * a_proj1_b = nullptr;
    struct ggml_tensor * a_proj2_w = nullptr; // [1024,896]
    struct ggml_tensor * a_proj2_b = nullptr;

    std::vector<qwen3_asr_audio_layer> asr_audio_layers;

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
