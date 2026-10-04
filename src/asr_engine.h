#pragma once

// ============================================================
// Qwen3-ASR 最小阻塞推理 API（asr/v0.3）
//
// 首个无需 Python 的完整离线转写入口：输入单声道 16 kHz float32 PCM
// （示例层支持 PCM16 WAV，按 /32768 转换），输出语言与转写文字。
//
// 链路（全部在 C++ 内完成）：
//   PCM → Log-Mel（src/log_mel.*，纯 DSP）
//       → 音频 Encoder（src/audio_encoder.*，ggml 音频塔）
//       → ASR Prompt 构造 + 分词（audio_pad 按 A(T) 展开）
//       → audio_pad 位置特征替换 → 混合 Embedding prefill
//       → token ID 增量贪心 decode（默认贪心、无重复惩罚）
//       → 协议解析（language 前缀与 <asr_text> 标记剥离）
//
// 生成约束（doc/asr/design_asr.md §4.2 第 5 条）：
//   - 停止条件 = EOS 集合 asr.eos_token_ids（<|im_end|>/<|endoftext|>），
//     达到 max_gen_tokens（LENGTH），或 KV 容量耗尽（CONTEXT_FULL）；
//   - 指定语言模式在 assistant 段预置 `language X<asr_text>`，
//     自动语言模式由模型自行输出协议头，响应解析后返回 language 字段。
//
// 线程模型：与 Frostfall::InferenceEngine 对齐——Init/Transcribe 互斥串行；
// 推理期间无回调、无跨线程入口（取消、流式分别属 asr/v0.4+）。
//
// 输入边界（工程边界，非模型能力上限）：单请求、单音频、不超过
// max_audio_seconds；空输入、非 16 kHz、NaN/Inf、超长一律显式报错，
// 不静默截断、不静默改采样率标签。
// ============================================================

#include "graph.h"
#include "kv_cache.h"
#include "log_mel.h"
#include "model.h"
#include "tokenizer.h"

#include "ggml-alloc.h"

#include <cstdint>
#include <string>
#include <vector>

// 引擎配置（Init 用；样例 JSON 见 examples/asr/api_test/asr_conf_v03.json）。
struct qwen3_asr_engine_config {
    std::string model_path;               // ASR GGUF 路径（必填，架构须为 qwen3-asr）
    int32_t n_ctx = 0;                    // KV 容量；0 = min(4096, n_ctx_train)
    int32_t max_audio_seconds = 8;        // 单请求音频时长上限（秒，对应 A ≤ 104 单窗约束）
    int32_t max_gen_tokens = 64;          // 请求未指定生成上限时的默认值
};

// 一次阻塞转写请求。pcm 归属调用方，仅在本调用内读取。
struct qwen3_asr_request {
    const float * pcm = nullptr;   // 单声道 float32 PCM（建议已归一到 [-1,1]）
    int32_t n_samples = 0;         // 采样数
    int32_t sample_rate = 16000;  // 必须等于 GGUF asr.mel.sampling_rate（16k）

    std::string context;       // 可选：system 段附加文本（转写上下文/风格提示）
    std::string language;      // 空 = 自动语言；非空 = 指定语言（如 "Chinese"）
    int32_t max_gen_tokens = 0; // 生成 token 上限；<=0 用引擎配置默认

    bool want_debug = false;   // 为 true 时响应携带 mel / 长度 / 原始输出等调试量
};

// 生成结束原因：区分“识别完成”与“生成截断”。
enum qwen3_asr_finish_reason {
    QWEN3_ASR_FINISH_EOS = 0,       // 正常结束：生成到 EOS 集合
    QWEN3_ASR_FINISH_LENGTH,        // 截断：达到请求 max_gen_tokens
    QWEN3_ASR_FINISH_CONTEXT_FULL,  // 截断：KV 容量耗尽（S+生成已达 n_ctx）
};

inline const char * qwen3_asr_finish_reason_str(qwen3_asr_finish_reason r) {
    switch (r) {
        case QWEN3_ASR_FINISH_EOS:          return "eos";
        case QWEN3_ASR_FINISH_LENGTH:        return "length";
        case QWEN3_ASR_FINISH_CONTEXT_FULL: return "context_full";
    }
    return "?";
}

struct qwen3_asr_response {
    bool ok = false;
    std::string error;                 // ok=false 时的失败原因（输入错误 / 内部错误）

    std::string text;                  // 转写正文（已剥离 language 协议头与 <asr_text>）
    std::string language;              // 指定模式 = 请求值；自动模式 = 模型协议输出（如 "Chinese"）
    qwen3_asr_finish_reason finish_reason = QWEN3_ASR_FINISH_EOS;

    // ---- 调试输出（request.want_debug=true 时填充） ----
    int32_t T = 0;                     // 有效 Mel 帧数
    int32_t A = 0;                     // 音频位置数
    int32_t S = 0;                     // prefill token 数
    int32_t n_gen = 0;                 // 生成 token 数（含 EOS）
    std::vector<int32_t> gen_ids;      // 生成的 token id（含 EOS/末 token）
    std::string raw;                   // 原始生成文本（skip_special 解码，含协议头）
    std::vector<float> mel;            // Log-Mel [n_mels, T]（请求失败于前处理阶段则为空）
    float log_mel_max = 0.0f;
};

class qwen3_asr_engine {
public:
    qwen3_asr_engine() = default;
    ~qwen3_asr_engine();
    qwen3_asr_engine(const qwen3_asr_engine &) = delete;
    qwen3_asr_engine & operator=(const qwen3_asr_engine &) = delete;

    // 初始化（加载 GGUF 模型 + tokenizer + KV）。失败返回 false（原因走 stderr 日志）。
    bool Init(const qwen3_asr_engine_config & config);

    // 从 JSON 配置文件初始化（字段同 qwen3_asr_engine_config）。
    bool InitFromJsonFile(const std::string & path);

    // 阻塞转写：一次音频请求，返回语言与文字。每请求入口重置 Decoder KV，重复调用不串状态。
    // 输入校验失败（采样率不符/超长/空/NaN）时 ok=false 且不产生任何计算副作用。
    qwen3_asr_response Transcribe(const qwen3_asr_request & request);

    // 只读访问（诊断用）
    bool IsInit() const { return model_.backend != nullptr; }
    int32_t NCtx() const { return kv_.n_ctx; }

private:
    bool eval_token_graph(const int32_t * tokens, const float * embd_in,
                          int32_t n_tokens, int32_t n_past,
                          std::vector<float> & logits_out);

    qwen3_model       model_;
    qwen3_kv_cache    kv_;
    qwen3_tokenizer   tok_;
    ggml_gallocr_t    allocr_ = nullptr;
    qwen3_asr_engine_config config_;
};
