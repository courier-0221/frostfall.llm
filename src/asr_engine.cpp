#include "asr_engine.h"

#include "audio_encoder.h"
#include "log.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>

// ============================ 请求内的图容量 ============================
// AudioEncoder / Decoder 构图共用 quota；代码与 asr_align_v02 一致（16k 节点，实测 < 12k）。
static constexpr int kMaxNodes = 16384;

// asr/v0.4 内部计时（wall-clock 秒）。now() 为入口相对时刻基点。
namespace {
using asr_clock = std::chrono::steady_clock;
inline double sec_since(asr_clock::time_point t0) {
    return std::chrono::duration<double>(asr_clock::now() - t0).count();
}
} // namespace

qwen3_asr_engine::~qwen3_asr_engine() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

namespace {

// 取单个 token 的文本 Embedding（tok_embd F16 -> F32），写入 dst[n_embd]。
void token_embd(const qwen3_model & model, int32_t id, float * dst) {
    const int32_t n_embd = model.hparams.n_embd;
    std::vector<ggml_fp16_t> row((size_t) n_embd);
    ggml_backend_tensor_get(model.tok_embd, row.data(),
                            (size_t) id * n_embd * sizeof(ggml_fp16_t),
                            (size_t) n_embd * sizeof(ggml_fp16_t));
    for (int32_t i = 0; i < n_embd; ++i) dst[i] = ggml_fp16_to_fp32(row[i]);
}

// 构造 ASR Prompt（官方 chat_template，单音频，不加 Thinking 前缀）：
//   <|im_start|>system\n{context}<|im_end|>\n
//   <|im_start|>user\n<|audio_start|>{A 个 <|audio_pad|>}<|audio_end|><|im_end|>\n
//   <|im_start|>assistant\n
//   [language {lang}<asr_text>]   （指定语言模式才追加）
std::string build_prompt(int32_t audio_pad_count, const std::string & language,
                         const std::string & context) {
    std::string pad;
    pad.reserve((size_t) audio_pad_count * 13);
    for (int32_t i = 0; i < audio_pad_count; ++i) pad += "<|audio_pad|>";
    std::string prompt = "<|im_start|>system\n" + context + "<|im_end|>\n"
                       + "<|im_start|>user\n<|audio_start|>" + pad + "<|audio_end|><|im_end|>\n"
                       + "<|im_start|>assistant\n";
    if (!language.empty()) prompt += "language " + language + "<asr_text>";
    return prompt;
}

// 解析生成协议：生成序列中 `<asr_text>`（USER_DEFINED token）之前的 token 是协议头
// （自动语言模式模型先吐 `language X`），之后才是正文。返回正文 text 与协议声明的
// 语言；协议缺失时 language 为空、text 为整段生成正文。
void parse_protocol(const qwen3_tokenizer & tok, const std::vector<int32_t> & gen,
                    std::string & language, std::string & text) {
    auto it = tok.token_to_id.find("<asr_text>");
    const int32_t marker = (it != tok.token_to_id.end()) ? it->second : -1;

    if (marker >= 0) {
        size_t p = 0;
        while (p < gen.size() && gen[p] != marker) ++p;
        if (p < gen.size()) {
            std::string head = tok.decode(std::vector<int32_t>(gen.begin(), gen.begin() + (long) p),
                                          /*skip_special=*/true);
            text = tok.decode(std::vector<int32_t>(gen.begin() + (long) (p + 1), gen.end()),
                              /*skip_special=*/true);
            // 去掉 head 尾部空白后应形如 `language Chinese`
            while (!head.empty() && (head.back() == '\n' || head.back() == ' ')) head.pop_back();
            if (head.rfind("language ", 0) == 0) {
                language = head.substr(std::strlen("language "));
                while (!language.empty() && (language.front() == ' ' || language.front() == '\n'))
                    language.erase(language.begin());
            }
            return;
        }
    }
    // 无协议标记：整段视为正文
    language.clear();
    text = tok.decode(gen, /*skip_special=*/true);
}

// UTF-8 续长度（0 = ASCII / 非法首字节按 1 字节消费，不向前抛错）。
inline int utf8_len(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// 取前缀中完整 UTF-8 序列的最大字节数：末尾不完整序列字节留给下一次回调；
// 全部完整时返回 s.size()。max_seq 上界防止任意长续字节。
inline size_t utf8_complete_prefix(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const int len = utf8_len((unsigned char) s[i]);
        if (len > 5) return i;                       // 非法首字节，保守截断
        if (i + (size_t) len > s.size()) break;      // 尾部不完整序列
        for (int j = 1; j < len; ++j)                // 校验续字节 0b10xxxxxx
            if ((s[i + (size_t) j] & 0xC0) != 0x80) break;
        i += (size_t) len;
    }
    return i;
}

} // namespace

// ============================ Init ============================

bool qwen3_asr_engine::Init(const qwen3_asr_engine_config & config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (IsInit()) {
        LOG(ERROR) << "asr_engine: already initialized; skip re-init"; // 重复 Init 保持首次状态
        return true;
    }
    if (config.model_path.empty()) {
        LOG(ERROR) << "asr_engine: config.model_path is empty";
        return false;
    }
    if (config.max_audio_seconds <= 0 || config.max_gen_tokens <= 0 || config.prefill_batch_size < 0) {
        LOG(ERROR) << "asr_engine: invalid config (max_audio_seconds/max_gen_tokens must be > 0; "
                      "prefill_batch_size must be >= 0)";
        return false;
    }
    config_ = config;

    if (!qwen3_model_load(config_.model_path, model_)) {
        LOG(ERROR) << "asr_engine: failed to load model from " << config_.model_path;
        return false;
    }
    if (!model_.is_asr) {
        LOG(ERROR) << "asr_engine: model architecture is not qwen3-asr: " << config_.model_path;
        return false;
    }
    if (!tok_.load(config_.model_path)) {
        LOG(ERROR) << "asr_engine: failed to load tokenizer from " << config_.model_path;
        return false;
    }

    int32_t n_ctx = config_.n_ctx > 0 ? config_.n_ctx : 4096;
    if (model_.hparams.n_ctx_train > 0) n_ctx = std::min(n_ctx, model_.hparams.n_ctx_train);
    if (!kv_.init(model_, n_ctx)) {
        LOG(ERROR) << "asr_engine: failed to init kv cache (n_ctx=" << n_ctx << ")";
        return false;
    }
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_.backend));
    if (!allocr_) {
        LOG(ERROR) << "asr_engine: failed to create graph allocator";
        return false;
    }
    return true;
}

bool qwen3_asr_engine::InitFromJsonFile(const std::string & path) {
    // 与 Frostfall::InferenceEngine::InitFromJsonFile 相同的解析约定：
    // 缺失字段保留结构体默认值。
    try {
        std::ifstream f(path);
        if (!f) {
            LOG(ERROR) << "asr_engine: cannot open config file " << path;
            return false;
        }
        const auto j = nlohmann::json::parse(f);
        qwen3_asr_engine_config cfg;
        if (j.contains("model_path"))      cfg.model_path = j["model_path"].get<std::string>();
        if (j.contains("n_ctx"))            cfg.n_ctx = j["n_ctx"].get<int32_t>();
        if (j.contains("max_audio_seconds")) cfg.max_audio_seconds = j["max_audio_seconds"].get<int32_t>();
        if (j.contains("max_gen_tokens"))   cfg.max_gen_tokens = j["max_gen_tokens"].get<int32_t>();
        if (j.contains("prefill_batch_size")) cfg.prefill_batch_size = j["prefill_batch_size"].get<int32_t>();
        return Init(cfg);
    } catch (const std::exception & e) {
        LOG(ERROR) << "asr_engine: failed to parse config " << path << ": " << e.what();
        return false;
    }
}

// ============================ 单步前向（prefill / 单 token decode 共用） ============================

bool qwen3_asr_engine::eval_token_graph(const int32_t * tokens, const float * embd_in,
                                        int32_t n_tokens, int32_t n_past,
                                        std::vector<float> & logits_out) {
    struct ggml_init_params gp = {
        /*.mem_size =*/ ggml_tensor_overhead() * kMaxNodes * 4 + ggml_graph_overhead_custom(kMaxNodes, false),
        /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true,
    };
    struct ggml_context * ctx = ggml_init(gp);
    if (!ctx) return false;

    qwen3_graph_params gparams;
    gparams.n_tokens = n_tokens;
    gparams.n_past   = n_past;
    gparams.max_nodes = kMaxNodes;
    gparams.logits_last_only  = true;
    gparams.want_layer_outputs = false;

    struct ggml_tensor * embd_t = nullptr;
    if (embd_in) {
        embd_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model_.hparams.n_embd, n_tokens);
        ggml_set_name(embd_t, QWEN3_TENSOR_NAME_EMBD);
        ggml_set_input(embd_t);
        gparams.embd = embd_t;
    }

    struct ggml_cgraph * gf = qwen3_build_graph(ctx, model_, kv_, gparams);
    if (!gf) { ggml_free(ctx); return false; }
    ggml_gallocr_alloc_graph(allocr_, gf);

    if (embd_in) {
        ggml_backend_tensor_set(embd_t, embd_in, 0,
                                (size_t) n_tokens * model_.hparams.n_embd * sizeof(float));
    } else {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_TOKENS),
                                tokens, 0, (size_t) n_tokens * sizeof(int32_t));
    }
    std::vector<int32_t> pos((size_t) n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) pos[(size_t) i] = n_past + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_POS),
                            pos.data(), 0, (size_t) n_tokens * sizeof(int32_t));

    const int32_t n_kv = n_past + n_tokens;
    std::vector<float> mask((size_t) n_tokens * n_kv, 0.0f);
    for (int32_t q = 0; q < n_tokens; ++q)
        for (int32_t k = n_past + q + 1; k < n_kv; ++k)
            mask[(size_t) q * n_kv + k] = -INFINITY;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_MASK),
                            mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_graph_compute(model_.backend, gf);

    logits_out.resize((size_t) model_.hparams.n_vocab);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_LOGITS),
                            logits_out.data(), 0,
                            (size_t) model_.hparams.n_vocab * sizeof(float));
    kv_.n_past = n_kv;
    ggml_free(ctx);
    return true;
}

// 取消路径统一收口：ok=true + FINISH_CANCELLED + 取消时阶段 + 部分统计。
// 取消不是错误：不写 error/err；text 为空（取消点在音频段/prefill 时无正文可给）。
// t_preproc/t_encoder 在此按阶段推断（取消点之前的阶段计为已耗时）。
static qwen3_asr_response asr_cancel_return(qwen3_asr_response & resp,
                                            std::chrono::steady_clock::time_point t_start,
                                            const qwen3_asr_request & req,
                                            qwen3_asr_stage stage_at_cancel) {
    resp.ok = true;
    resp.err = QWEN3_ASR_ERR_NONE;
    resp.error.clear();
    resp.finish_reason = QWEN3_ASR_FINISH_CANCELLED;
    resp.stage = stage_at_cancel;
    resp.stats.t_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    if (req.n_samples > 0 && req.sample_rate > 0)
        resp.stats.rtf = resp.stats.t_total / ((double) req.n_samples / req.sample_rate);
    return resp;
}

// ============================ Transcribe ============================

int32_t qwen3_asr_engine::MaxAudioPositions() const {
    return qwen3_asr_max_audio_positions(model_);
}

qwen3_asr_response qwen3_asr_engine::Transcribe(const qwen3_asr_request & req,
                                                qwen3_asr_stream_callback callback) {
    std::lock_guard<std::mutex> lock(mutex_); // Init/Transcribe 互斥串行；并发调用排队
    const auto t_start = asr_clock::now();
    qwen3_asr_response resp;
    resp.stage = QWEN3_ASR_STAGE_INPUT;

    auto cancelled = [&]() {
        return req.cancel != nullptr && req.cancel->load(std::memory_order_relaxed);
    };

    auto fail = [&](qwen3_asr_error err, qwen3_asr_stage stage, const std::string & msg) {
        resp.ok = false;
        resp.err = err;
        resp.stage = stage;
        resp.error = msg;
        resp.stats.t_total = sec_since(t_start);
        LOG(ERROR) << "asr_engine::Transcribe: " << qwen3_asr_error_str(err) << ": " << msg;
        return resp;
    };

    // ---- 输入校验（显式报错，不静默截断/改标签） ----
    if (!IsInit()) return fail(QWEN3_ASR_ERR_NOT_INIT, QWEN3_ASR_STAGE_INPUT, "engine is not initialized");
    if (!req.pcm && req.n_samples > 0) return fail(QWEN3_ASR_ERR_BAD_INPUT, QWEN3_ASR_STAGE_INPUT, "pcm is null");
    if (req.n_samples <= 0)
        return fail(QWEN3_ASR_ERR_BAD_INPUT, QWEN3_ASR_STAGE_INPUT,
                    "empty input: n_samples=" + std::to_string(req.n_samples));
    if (req.sample_rate != model_.asr_hparams.mel_sampling_rate)
        return fail(QWEN3_ASR_ERR_BAD_INPUT, QWEN3_ASR_STAGE_INPUT,
                    "unsupported sample_rate=" + std::to_string(req.sample_rate) +
                    " (this engine only accepts " +
                    std::to_string(model_.asr_hparams.mel_sampling_rate) +
                    " Hz; resample before calling)");
    const int64_t max_samples = (int64_t) config_.max_audio_seconds * req.sample_rate;
    if (req.n_samples > max_samples)
        return fail(QWEN3_ASR_ERR_AUDIO_TOO_LONG, QWEN3_ASR_STAGE_INPUT,
                    "audio too long: n_samples=" + std::to_string(req.n_samples) + " ("
                    + std::to_string(req.n_samples / req.sample_rate) + "s) > limit "
                    + std::to_string(config_.max_audio_seconds) + "s");
    for (int32_t i = 0; i < req.n_samples; ++i)
        if (!std::isfinite(req.pcm[i]))
            return fail(QWEN3_ASR_ERR_BAD_INPUT, QWEN3_ASR_STAGE_INPUT,
                        "pcm contains NaN/Inf at sample " + std::to_string(i));
    if (req.max_gen_tokens < 0) return fail(QWEN3_ASR_ERR_BAD_INPUT, QWEN3_ASR_STAGE_INPUT, "max_gen_tokens must be >= 0");
    if (req.prefill_batch_size < 0) return fail(QWEN3_ASR_ERR_BAD_INPUT, QWEN3_ASR_STAGE_INPUT, "prefill_batch_size must be >= 0");
    const int32_t max_gen = req.max_gen_tokens > 0 ? req.max_gen_tokens : config_.max_gen_tokens;
    const int32_t batch_size = req.prefill_batch_size > 0 ? req.prefill_batch_size
                                   : (config_.prefill_batch_size > 0 ? config_.prefill_batch_size : 0);

    // ---- 1. Log-Mel（纯 DSP，输入 PCM） ----
    double preproc_seconds = 0.0;
    qwen3_asr_mel_hparams mp;
    mp.sampling_rate   = model_.asr_hparams.mel_sampling_rate;
    mp.n_fft           = model_.asr_hparams.mel_n_fft;
    mp.hop_length      = model_.asr_hparams.mel_hop_length;
    mp.feature_size    = model_.asr_hparams.mel_feature_size;
    mp.padding_side    = model_.asr_hparams.mel_padding_side;

    qwen3_asr_log_mel_result mel_res;
    std::string mel_err;
    {
        const auto t_mel = asr_clock::now();
        if (!qwen3_asr_log_mel(mp, req.pcm, req.n_samples, mel_res, &mel_err))
            return fail(QWEN3_ASR_ERR_INTERNAL, QWEN3_ASR_STAGE_PREPROC, "log-mel failed: " + mel_err);
        preproc_seconds = sec_since(t_mel);
    }
    const int32_t T = mel_res.T;
    resp.T = T; // 长度信息始终返回，便于诊断
    if (req.want_debug) {
        resp.mel = std::move(mel_res.mel);
        resp.log_mel_max = mel_res.log_mel_max;
    }
    resp.stage = QWEN3_ASR_STAGE_PREPROC;

    // ---- 2. 音频 Encoder（一次编码；跨窗口输入按官方 cu_seqlens 块对角双向处理） ----
    double encoder_seconds = 0.0;
    std::vector<float> feats;
    {
        const auto t_enc = asr_clock::now();
        std::vector<float> mel_vec;
        if (req.want_debug) mel_vec = resp.mel; // debug 需要保留矩阵，此处另取副本
        else               mel_vec = std::move(mel_res.mel);
        qwen3_asr_audio_result audio;
        if (!qwen3_asr_encode_audio(model_, mel_vec.data(), T, /*want_intermediates=*/false, audio,
                                    [&]() { return cancelled(); })) {
            if (cancelled())
                return asr_cancel_return(resp, t_start, req, QWEN3_ASR_STAGE_ENCODER);
            return fail(QWEN3_ASR_ERR_INTERNAL, QWEN3_ASR_STAGE_ENCODER,
                        "audio encoder failed (T=" + std::to_string(T) + ")");
        }
        feats = std::move(audio.features);
        encoder_seconds = sec_since(t_enc);
        if (cancelled()) return asr_cancel_return(resp, t_start, req, QWEN3_ASR_STAGE_ENCODER);
    }
    resp.stage = QWEN3_ASR_STAGE_ENCODER;
    const int32_t A = (int32_t) (feats.size() / model_.hparams.n_embd);
    if (req.want_debug) resp.A = A;

    // ---- 3. ASR Prompt / 分词 / 融合 Embedding ----
    const std::string prompt = build_prompt(A, req.language, req.context);
    std::vector<int32_t> ids = tok_.encode(prompt);
    const int32_t S = (int32_t) ids.size();
    const int32_t audio_pad_id = model_.asr_hparams.audio_token_id;
    size_t pad_count = 0;
    for (int32_t id : ids) if (id == audio_pad_id) ++pad_count;
    if ((int32_t) pad_count != A)
        return fail(QWEN3_ASR_ERR_INTERNAL, QWEN3_ASR_STAGE_ENCODER,
                    "internal error: prompt audio_pad count " + std::to_string(pad_count) +
                    " != audio positions A=" + std::to_string(A));
    if (req.want_debug) {
        resp.A = A;
        resp.S = S;
    }

    const int32_t n_embd = model_.hparams.n_embd;
    std::vector<float> fused((size_t) S * n_embd);
    {
        size_t ai = 0;
        for (int32_t s = 0; s < S; ++s) {
            float * row = fused.data() + (size_t) s * n_embd;
            if (ids[(size_t) s] == audio_pad_id) {
                std::memcpy(row, feats.data() + ai * (size_t) n_embd,
                            (size_t) n_embd * sizeof(float));
                ++ai;
            } else {
                token_embd(model_, ids[(size_t) s], row);
            }
        }
    }

    // ---- 容量检查：prefill + 至少 1 个生成都放不进 KV 时显式报错 ----
    if (S + 1 > kv_.n_ctx)
        return fail(QWEN3_ASR_ERR_CONTEXT_INSUFFICIENT, QWEN3_ASR_STAGE_ENCODER,
                    "context insufficient: prefill " + std::to_string(S) +
                    " tokens + 1 gen > n_ctx " + std::to_string(kv_.n_ctx) +
                    " (audio may exceed the KV window quota)");

    // ---- 4. 分批 prefill ----
    // 批内完整因果（位置/mask/n_past 对每批从 kv_.n_past 顺延），末位 logits 与
    // 单批一致（因果 mask 下前面批次的末位 logits 会被后续批次覆盖，最终只保留 S-1 位）。
    kv_.n_past = 0; // 每请求重置：重复调用不串 KV
    resp.stage = QWEN3_ASR_STAGE_PREFILL;
    std::vector<float> logits;
    {
        double t_prefill = 0.0;
        int32_t n_batches = 0;
        const int32_t bs = batch_size > 0 ? batch_size : S; // <=0 = 单批（v0.3 行为）
        size_t peak = 0;
        for (int32_t p0 = 0; p0 < S; p0 += bs) {
            if (cancelled()) return asr_cancel_return(resp, t_start, req, QWEN3_ASR_STAGE_PREFILL);
            const int32_t nb = std::min(bs, S - p0);
            const auto t_b = asr_clock::now();
            std::vector<float> logits_tmp;
            if (p0 + nb >= S) { // 末批：保留末位 logits
                if (!eval_token_graph(nullptr, fused.data() + (size_t) p0 * n_embd, nb, kv_.n_past, logits))
                    return fail(QWEN3_ASR_ERR_INTERNAL, QWEN3_ASR_STAGE_PREFILL, "prefill graph failed");
            } else {
                if (!eval_token_graph(nullptr, fused.data() + (size_t) p0 * n_embd, nb, kv_.n_past, logits_tmp))
                    return fail(QWEN3_ASR_ERR_INTERNAL, QWEN3_ASR_STAGE_PREFILL, "prefill graph failed");
            }
            t_prefill += sec_since(t_b);
            ++n_batches;
            peak = std::max(peak, ggml_gallocr_get_buffer_size(allocr_, 0));
        }
        resp.stats.t_prefill = t_prefill;
        resp.stats.n_prefill_batches = n_batches;
        resp.stats.peak_galloc_bytes = peak;
    }

    const int32_t n_vocab = model_.hparams.n_vocab;
    auto argmax = [&](int32_t & out_id) {
        out_id = 0;
        float bv = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v)
            if (logits[(size_t) v] > bv) { bv = logits[(size_t) v]; out_id = v; }
    };

    // ---- 5. 逐 token 贪心 decode + 文本流式 ----
    // 流式缓冲规则：
    //   - <asr_text> 之前的生成 token 属协议段（自动语言模式为 `language X`），
    //     全部缓冲在 protocol_buf，不回调文本；解析出语言后回调 STREAM_LANG 一次。
    //   - 正文增量经 pending 缓冲，只在 UTF-8 完整序列边界回调 STREAM_TEXT；
    //     callback 返回 false = 调用方取消（与置位 req.cancel 等价）。
    std::vector<int32_t> gen;
    resp.finish_reason = QWEN3_ASR_FINISH_EOS;
    double t_decode = 0.0, t_first_model = -1.0, t_first_text = -1.0;
    int32_t n_steps = 0;

    const int32_t asr_text_id = [&]() {
        auto it = tok_.token_to_id.find("<asr_text>");
        return it != tok_.token_to_id.end() ? it->second : -1;
    }();
    bool in_protocol = true;   // 自动语言模式先吐协议头；指定模式在 prompt 预置，生成直接进正文
    if (!req.language.empty()) in_protocol = false;
    std::string protocol_buf, pending, emitted;

    auto flush_stream = [&]() -> bool {
        if (!callback || pending.empty()) return true;
        const size_t complete = utf8_complete_prefix(pending);
        if (complete == 0) return true;
        const std::string piece = pending.substr(0, complete);
        pending.erase(0, complete);
        if (piece.empty()) return true;
        if (t_first_text < 0.0) t_first_text = sec_since(t_start);
        emitted += piece;
        return callback(QWEN3_ASR_STREAM_TEXT, piece);
    };

    for (int32_t step = 0;; ++step) {
        if (cancelled()) { resp.finish_reason = QWEN3_ASR_FINISH_CANCELLED; break; }
        if (step > 0) {
            const auto t_d = asr_clock::now();
            if (!eval_token_graph(&gen.back(), nullptr, 1, kv_.n_past, logits))
                return fail(QWEN3_ASR_ERR_INTERNAL, QWEN3_ASR_STAGE_DECODE, "decode step graph failed");
            t_decode += sec_since(t_d);
            ++n_steps;
            resp.stage = QWEN3_ASR_STAGE_DECODE;
        }
        int32_t best = 0;
        argmax(best);
        if (step == 0) t_first_model = sec_since(t_start);
        gen.push_back(best);

        bool is_eos = false;
        for (int32_t e : model_.asr_hparams.eos_token_ids) if (best == e) { is_eos = true; break; }

        // 正文增量先回调/缓冲，再检查停止条件（EOS 本身不进入正文）。
        if (!is_eos) {
            const std::string piece = tok_.decode({best}, /*skip_special=*/false);
            if (in_protocol) {
                protocol_buf += piece;
                const size_t pos = protocol_buf.find("<asr_text>");
                if (pos != std::string::npos) {
                    in_protocol = false;
                    std::string head = protocol_buf.substr(0, pos);
                    // 去首尾空白：协议头应形如 `language X`；language None = 无语音
                    std::string lang_clean;
                    size_t b = 0, e = head.size();
                    while (b < e && (head[b] == ' ' || head[b] == '\n')) ++b;
                    while (e > b && (head[e - 1] == ' ' || head[e - 1] == '\n')) --e;
                    lang_clean = head.substr(b, e - b);
                    // 语言协议 token 不计入首可见文字（§5.4：首可见文字 = 首个转写正文）
                    if (callback && !lang_clean.empty()) {
                        if (!callback(QWEN3_ASR_STREAM_LANG, lang_clean)) {
                            resp.finish_reason = QWEN3_ASR_FINISH_CANCELLED;
                            break;
                        }
                    }
                }
            } else if (best != asr_text_id) { // 指定模式首 token 可能仍是 <asr_text>：吞掉
                pending += piece;
                if (!flush_stream()) { resp.finish_reason = QWEN3_ASR_FINISH_CANCELLED; break; }
            }
        }

        if (is_eos) break;
        if ((int32_t) gen.size() >= max_gen) { resp.finish_reason = QWEN3_ASR_FINISH_LENGTH; break; }
        if (kv_.n_past + 1 > kv_.n_ctx) { resp.finish_reason = QWEN3_ASR_FINISH_CONTEXT_FULL; break; }
    }

    // ---- 6. 文本与协议解析 ----
    const std::string raw = tok_.decode(gen, /*skip_special=*/false);
    std::string parsed_lang, text;
    parse_protocol(tok_, gen, parsed_lang, text);

    // 阻塞模式：text 来自解析。流式模式：正常结束 emitted 与 text 一致（生成顺序
    // 即 flush 顺序）；取消时只保留已确认回调的正文（pending 中的不完整/未 flush
    // 尾段不并入），便于上层展示“取消前已收到的文字”。
    resp.text = callback ? (emitted.empty() && pending.empty() ? text
                          : (resp.finish_reason == QWEN3_ASR_FINISH_CANCELLED ? emitted
                          : (emitted + pending))) : text;
    if (callback && resp.finish_reason != QWEN3_ASR_FINISH_CANCELLED) {
        // 正常结束：与解析结果一致性由调用方校验（两者应逐字节一致）
        ;
    }

    resp.ok = true;
    resp.language = !req.language.empty() ? req.language : parsed_lang;
    resp.n_gen = (int32_t) gen.size();
    if (resp.finish_reason != QWEN3_ASR_FINISH_CANCELLED) resp.stage = QWEN3_ASR_STAGE_DONE;
    // 取消时保留 stage = 取消时所处阶段（ENCODER/PREFILL/DECODE）

    // ---- 7. 性能统计 ----
    const auto t_end_total = asr_clock::now();
    resp.stats.t_total = std::chrono::duration<double>(t_end_total - t_start).count();
    resp.stats.t_preproc = preproc_seconds;
    resp.stats.t_encoder = encoder_seconds;
    resp.stats.t_decode = t_decode;
    resp.stats.n_decode_steps = n_steps;
    resp.stats.t_first_model_token = t_first_model;
    resp.stats.t_first_visible_text = t_first_text;
    resp.stats.rtf = resp.stats.t_total / ((double) req.n_samples / req.sample_rate);
    resp.stats.prefill_batch_size = batch_size > 0 ? std::min(batch_size, S) : S;
    if (req.want_debug) {
        resp.gen_ids = std::move(gen);
        resp.raw = raw;
    }
    (void) text;
    return resp;
}
