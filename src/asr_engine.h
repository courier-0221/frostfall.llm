#pragma once

// ============================================================
// Qwen3-ASR 完整离线推理 API（asr/v0.4）
//
// 在 v0.3 最小阻塞入口（pcm → Log-Mel → 音频 Encoder → prefill →
// 贪心 decode → 协议解析 → 语言 + 文字）之上，本版补齐应用接入能力：
//
//   1. 文本流式    Transcribe(cb)：完整音频处理后逐 token 回调正文
//                  （增量 UTF-8 严格边界、协议头缓冲不外泄；自动语言
//                  模式协议解析后先回调 language 一次）。
//   2. 取消        request.cancel 由其他线程置位（std::atomic）。检查
//                  边界：音频块、prefill 批次、decode step；取消返回
//                  部分结果 + FINISH_CANCELLED + 取消时处理阶段。不承诺
//                  打断单个 ggml 算子（一个 prefill 批/decode step 为
//                  最小打断粒度）。
//   3. 分批 prefill 混合 Embedding 按 prefill_batch_size 分批前向，
//                  切片/位置/mask/n_past 同步推进；批切片不改变因果
//                  语义与末位 logits，数值与单批一致。
//   4. 跨窗口音频  音频 Transformer 按官方 cu_seqlens 语义做块对角
//                  双向注意力（window_aftercnn = 13*(n_window_infer/100)
//                  = 104 位置；完整块凑满窗口、尾块收尾，窗口内双向、
//                  窗口间隔离）。单窗口输入结果与 v0.3 逐位一致；音频
//                  上限默认放宽至 30 秒。
//   5. 性能统计    resp.stats：前处理/Encoder/prefill/decode 耗时、首
//                  模型 token、首可见文字、RTF、峰值计算缓冲字节数。
//
// 错误协议：err = QWEN3_ASR_ERR_* 枚举（v0.3 只读 resp.error 的用法继续
// 可用；枚举面向程序分支）。
//
// 线程模型：Init/Transcribe 互斥串行（并发调用排队逐个执行）；重复调用
// 安全（每请求重置 KV）。回调约束：callback 只在推理线程内同步调用，
// 回调内不得再进入本引擎任何接口（重入即死锁）；callback 返回 false 表
// 示调用方主动取消（与置位 cancel 等价），引擎立即停止并返回部分结果。
//
// ============================================================

#include "graph.h"
#include "kv_cache.h"
#include "log_mel.h"
#include "model.h"
#include "tokenizer.h"

#include "ggml-alloc.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

// 引擎配置（Init 用；样例 JSON 见 examples/asr/api_test/asr_conf_v04.json）。
struct qwen3_asr_engine_config {
    std::string model_path;               // ASR GGUF 路径（必填，架构须为 qwen3-asr）
    int32_t n_ctx = 0;                    // KV 容量；0 = min(4096, n_ctx_train)
    int32_t max_audio_seconds = 30;       // 单请求音频时长上限（秒；跨窗口后默认放宽至 30）
    int32_t max_gen_tokens = 64;          // 请求未指定生成上限时的默认值
    int32_t prefill_batch_size = 128;     // 混合 Embedding 分批 prefill 的批大小；<=1 = 单批
};

// 一次转写请求（阻塞 / 文本流式共用）。pcm 归属调用方，仅在本调用内读取。
struct qwen3_asr_request {
    const float * pcm = nullptr;   // 单声道 float32 PCM（建议已归一到 [-1,1]）
    int32_t n_samples = 0;         // 采样数
    int32_t sample_rate = 16000;  // 必须等于 GGUF asr.mel.sampling_rate（16k）

    std::string context;       // 可选：system 段附加文本（转写上下文/风格提示）
    std::string language;      // 空 = 自动语言；非空 = 指定语言（如 "Chinese"）
    int32_t max_gen_tokens = 0; // 生成 token 上限；<=0 用引擎配置默认

    // 取消标志（asr/v0.4）。其他线程在推理期间置位 true，引擎在音频块、
    // prefill 批次、decode step 边界检查并尽快停止；nullptr = 不可取消。
    std::atomic<bool> * cancel = nullptr;

    bool want_debug = false;   // 为 true 时响应携带 mel / 长度 / 原始输出等调试量

    // 分批 prefill 批大小覆盖；0 = 用引擎配置 prefill_batch_size。
    int32_t prefill_batch_size = 0;
};

// 文本流式回调事件类型。
enum qwen3_asr_stream_event {
    QWEN3_ASR_STREAM_LANG = 0,  // 自动语言模式：协议头解析完成，正文 = 识别语言（一次性）
    QWEN3_ASR_STREAM_TEXT = 1,  // 正文增量（完整 UTF-8 序列，不含协议头）
};

// 文本流式回调：event + 当前增量片段。返回 false = 调用方取消（等价置位
// cancel），引擎立即停止并返回部分结果。回调在本引擎推理线程内同步调用；
// 回调内不得再进入本引擎任何接口（重入即死锁）。
using qwen3_asr_stream_callback = std::function<bool(qwen3_asr_stream_event event,
                                                     const std::string & piece)>;

// 生成结束原因：区分“识别完成”“生成截断”与“调用方取消”。
enum qwen3_asr_finish_reason {
    QWEN3_ASR_FINISH_EOS = 0,       // 正常结束：生成到 EOS 集合（<|im_end|>/）
    QWEN3_ASR_FINISH_LENGTH,        // 截断：达到请求 max_gen_tokens
    QWEN3_ASR_FINISH_CONTEXT_FULL,  // 截断：KV 容量耗尽（S+生成已达 n_ctx）
    QWEN3_ASR_FINISH_CANCELLED,     // v0.4：调用方请求取消，text 为部分结果
};

inline const char * qwen3_asr_finish_reason_str(qwen3_asr_finish_reason r) {
    switch (r) {
        case QWEN3_ASR_FINISH_EOS:          return "eos";
        case QWEN3_ASR_FINISH_LENGTH:       return "length";
        case QWEN3_ASR_FINISH_CONTEXT_FULL: return "context_full";
        case QWEN3_ASR_FINISH_CANCELLED:    return "cancelled";
    }
    return "?";
}

// 错误分类（程序分支用；resp.error 始终携带可读描述）。
enum qwen3_asr_error {
    QWEN3_ASR_ERR_NONE = 0,
    QWEN3_ASR_ERR_NOT_INIT,           // 引擎未初始化
    QWEN3_ASR_ERR_BAD_INPUT,          // 输入校验失败（空/NaN/采样率/参数非法）
    QWEN3_ASR_ERR_AUDIO_TOO_LONG,     // 超过 max_audio_seconds
    QWEN3_ASR_ERR_WINDOW_FULL,        // 音频超出窗口容量配额（非重叠逐段处理仍放不下）
    QWEN3_ASR_ERR_CONTEXT_INSUFFICIENT, // prefill+生成放不进 KV（经逐段处理后仍不足）
    QWEN3_ASR_ERR_INTERNAL,           // 内部错误（构图/编码/协议异常）
};

inline const char * qwen3_asr_error_str(qwen3_asr_error e) {
    switch (e) {
        case QWEN3_ASR_ERR_NONE:                 return "none";
        case QWEN3_ASR_ERR_NOT_INIT:             return "not_init";
        case QWEN3_ASR_ERR_BAD_INPUT:            return "bad_input";
        case QWEN3_ASR_ERR_AUDIO_TOO_LONG:       return "audio_too_long";
        case QWEN3_ASR_ERR_WINDOW_FULL:          return "window_full";
        case QWEN3_ASR_ERR_CONTEXT_INSUFFICIENT: return "context_insufficient";
        case QWEN3_ASR_ERR_INTERNAL:             return "internal";
    }
    return "?";
}

// 请求处理到哪个阶段被取消/失败（取消语义的一部分）。
enum qwen3_asr_stage {
    QWEN3_ASR_STAGE_INPUT = 0,   // 输入校验
    QWEN3_ASR_STAGE_PREPROC,     // Log-Mel 前处理
    QWEN3_ASR_STAGE_ENCODER,     // 音频塔编码
    QWEN3_ASR_STAGE_PREFILL,     // 混合 Embedding 分批 prefill
    QWEN3_ASR_STAGE_DECODE,      // 增量 decode
    QWEN3_ASR_STAGE_DONE,        // 完成
};

inline const char * qwen3_asr_stage_str(qwen3_asr_stage st) {
    switch (st) {
        case QWEN3_ASR_STAGE_INPUT:   return "input";
        case QWEN3_ASR_STAGE_PREPROC: return "preproc";
        case QWEN3_ASR_STAGE_ENCODER: return "encoder";
        case QWEN3_ASR_STAGE_PREFILL: return "prefill";
        case QWEN3_ASR_STAGE_DECODE:  return "decode";
        case QWEN3_ASR_STAGE_DONE:    return "done";
    }
    return "?";
}

// 性能统计（wall-clock，秒；asr/v0.4）。分段计时边界与 stage 一致；
// 前三段为完整音频的一次性开销，prefill/decode 为生成段累计耗时。
struct qwen3_asr_stats {
    double t_preproc   = 0.0;  // Log-Mel 前处理（含输入 PCM 扫描）
    double t_encoder   = 0.0;  // 音频塔（CNN + 18 层 Transformer + 投影）
    double t_prefill   = 0.0;  // 分批 prefill 累计
    double t_decode    = 0.0;  // decode step 累计（不含 prefill）
    double t_total     = 0.0;  // Transcribe 入口到出口总耗时
    double t_first_model_token = -1.0; // prefill 结束产出第 1 个生成 token 的时刻（相对入口；失败为 -1）
    double t_first_visible_text = -1.0;// 第 1 个正文可见文字回调/产出的时刻（相对入口；无正文为 -1；语言协议 token 不计入）
    double rtf         = 0.0;  // 总处理耗时 / 音频时长；音频时长 = n_samples/sample_rate
    int32_t n_prefill_batches = 0; // 实际 prefill 批数（含上下文切分续批）
    int32_t n_decode_steps = 0;    // 实际 decode step 数
    size_t peak_galloc_bytes = 0;  // 峰值计算缓冲（文本 Decoder 图分配器；音频塔与 KV 另计）
    int32_t prefill_batch_size = 0; // 本次请求生效的批大小
};

struct qwen3_asr_response {
    bool ok = false;
    qwen3_asr_error err = QWEN3_ASR_ERR_INTERNAL; // 错误分类（ok=false 时有效）
    std::string error;                 // ok=false 时的失败原因（输入错误 / 内部错误）

    std::string text;                  // 转写正文（已剥离 language 协议头与 <asr_text>）
    std::string language;              // 指定模式 = 请求值；自动模式 = 模型协议输出（如 "Chinese"）
    qwen3_asr_finish_reason finish_reason = QWEN3_ASR_FINISH_EOS;
    qwen3_asr_stage stage = QWEN3_ASR_STAGE_INPUT; // 完成时 = DONE；取消/失败时 = 所处阶段

    qwen3_asr_stats stats;             // 性能统计（asr/v0.4；ok=true 或取消/截断时填充）

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

    // 阻塞转写：一次音频请求，返回语言与文字。每请求入口重置 Decoder KV，重复
    // 调用不串状态；与 Transcribe(callback) 互斥串行。输入校验失败（采样率不符/
    // 超长/空/NaN）时 ok=false 且不产生任何计算副作用。传 callback 即文本流式：
    // 完整音频处理后逐 token 回调正文（协议头缓冲后才回调 language）。
    qwen3_asr_response Transcribe(const qwen3_asr_request & request,
                                  qwen3_asr_stream_callback callback = nullptr);

    // 只读访问（诊断用）
    bool IsInit() const { return model_.backend != nullptr; }
    int32_t NCtx() const { return kv_.n_ctx; }
    // 音频窗口配额（位置数）：单请求连续音频最多可用的音频 Transformer 位置数。
    int32_t MaxAudioPositions() const;

private:
    bool eval_token_graph(const int32_t * tokens, const float * embd_in,
                          int32_t n_tokens, int32_t n_past,
                          std::vector<float> & logits_out);

    std::mutex mutex_;                  // Init/Transcribe 互斥串行（并发调用排队）
    qwen3_model       model_;
    qwen3_kv_cache    kv_;
    qwen3_tokenizer   tok_;
    ggml_gallocr_t    allocr_ = nullptr;
    qwen3_asr_engine_config config_;
};
