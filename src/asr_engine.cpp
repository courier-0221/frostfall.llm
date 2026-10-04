#include "asr_engine.h"

#include "audio_encoder.h"
#include "log.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

// ============================ 请求内的图容量 ============================
// AudioEncoder / Decoder 构图共用 quota；代码与 asr_align_v02 一致（16k 节点，实测 < 12k）。
static constexpr int kMaxNodes = 16384;

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

} // namespace

// ============================ Init ============================

bool qwen3_asr_engine::Init(const qwen3_asr_engine_config & config) {
    if (config.model_path.empty()) {
        LOG(ERROR) << "asr_engine: config.model_path is empty";
        return false;
    }
    if (config.max_audio_seconds <= 0 || config.max_gen_tokens <= 0) {
        LOG(ERROR) << "asr_engine: invalid config (max_audio_seconds/max_gen_tokens must be > 0)";
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

// ============================ Transcribe ============================

qwen3_asr_response qwen3_asr_engine::Transcribe(const qwen3_asr_request & req) {
    qwen3_asr_response resp;

    auto fail = [&](const std::string & msg) {
        resp.ok = false;
        resp.error = msg;
        LOG(ERROR) << "asr_engine::Transcribe: " << msg;
        return resp;
    };

    // ---- 输入校验（显式报错，不静默截断/改标签） ----
    if (!IsInit()) return fail("engine is not initialized");
    if (!req.pcm && req.n_samples > 0) return fail("pcm is null");
    if (req.n_samples <= 0)
        return fail("empty input: n_samples=" + std::to_string(req.n_samples));
    if (req.sample_rate != model_.asr_hparams.mel_sampling_rate)
        return fail("unsupported sample_rate=" + std::to_string(req.sample_rate) +
                    " (this engine only accepts " +
                    std::to_string(model_.asr_hparams.mel_sampling_rate) +
                    " Hz; resample before calling)");
    const int64_t max_samples = (int64_t) config_.max_audio_seconds * req.sample_rate;
    if (req.n_samples > max_samples)
        return fail("audio too long: n_samples=" + std::to_string(req.n_samples) + " ("
                    + std::to_string(req.n_samples / req.sample_rate) + "s) > limit "
                    + std::to_string(config_.max_audio_seconds) + "s");
    for (int32_t i = 0; i < req.n_samples; ++i) {
        if (!std::isfinite(req.pcm[i]))
            return fail("pcm contains NaN/Inf at sample " + std::to_string(i));
    }
    if (req.max_gen_tokens < 0) return fail("max_gen_tokens must be >= 0");
    const int32_t max_gen = req.max_gen_tokens > 0 ? req.max_gen_tokens : config_.max_gen_tokens;

    // ---- 1. Log-Mel（纯 DSP，输入 PCM） ----
    qwen3_asr_mel_hparams mp;
    mp.sampling_rate   = model_.asr_hparams.mel_sampling_rate;
    mp.n_fft           = model_.asr_hparams.mel_n_fft;
    mp.hop_length      = model_.asr_hparams.mel_hop_length;
    mp.feature_size    = model_.asr_hparams.mel_feature_size;
    mp.padding_side    = model_.asr_hparams.mel_padding_side;

    qwen3_asr_log_mel_result mel_res;
    std::string mel_err;
    if (!qwen3_asr_log_mel(mp, req.pcm, req.n_samples, mel_res, &mel_err))
        return fail("log-mel failed: " + mel_err);
    const int32_t T = mel_res.T;
    resp.T = T; // 长度信息始终返回，便于诊断
    if (req.want_debug) {
        resp.mel = std::move(mel_res.mel);
        resp.log_mel_max = mel_res.log_mel_max;
    }

    // ---- 2. 音频 Encoder（一次编码，后续生成不再跑音频塔） ----
    std::vector<float> mel_vec;
    if (req.want_debug) mel_vec = resp.mel; // debug 需要保留矩阵，此处另取副本
    else               mel_vec = std::move(mel_res.mel);
    qwen3_asr_audio_result audio;
    if (!qwen3_asr_encode_audio(model_, mel_vec.data(), T, /*want_intermediates=*/false, audio))
        return fail("audio encoder failed (audio may exceed the single-window limit)");
    const int32_t A = audio.A;
    if (req.want_debug) resp.A = A;

    // ---- 3. ASR Prompt / 分词 / 融合 Embedding ----
    const std::string prompt = build_prompt(A, req.language, req.context);
    std::vector<int32_t> ids = tok_.encode(prompt);
    const int32_t S = (int32_t) ids.size();
    const int32_t audio_pad_id = model_.asr_hparams.audio_token_id;
    size_t pad_count = 0;
    for (int32_t id : ids) if (id == audio_pad_id) ++pad_count;
    if ((int32_t) pad_count != A)
        return fail("internal error: prompt audio_pad count " + std::to_string(pad_count) +
                    " != audio positions A=" + std::to_string(A));
    if (req.want_debug) resp.S = S;

    const int32_t n_embd = model_.hparams.n_embd;
    std::vector<float> fused((size_t) S * n_embd);
    {
        size_t ai = 0;
        for (int32_t s = 0; s < S; ++s) {
            float * row = fused.data() + (size_t) s * n_embd;
            if (ids[(size_t) s] == audio_pad_id) {
                std::memcpy(row, audio.features.data() + ai * (size_t) n_embd,
                            (size_t) n_embd * sizeof(float));
                ++ai;
            } else {
                token_embd(model_, ids[(size_t) s], row);
            }
        }
    }

    // ---- 容量检查：prefill + 至少 1 个生成都放不进 KV 时属输入错误 ----
    if (S + 1 > kv_.n_ctx)
        return fail("context insufficient: prefill " + std::to_string(S) +
                    " tokens + 1 gen > n_ctx " + std::to_string(kv_.n_ctx) +
                    " (audio may exceed the single-window limit)");

    // ---- 4. prefill + 增量贪心 decode ----
    kv_.n_past = 0; // 每请求重置：重复调用不串 KV

    std::vector<float> logits;
    if (!eval_token_graph(nullptr, fused.data(), S, 0, logits))
        return fail("prefill graph failed");

    const int32_t n_vocab = model_.hparams.n_vocab;
    auto argmax = [&](int32_t & out_id) {
        out_id = 0;
        float bv = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v)
            if (logits[(size_t) v] > bv) { bv = logits[(size_t) v]; out_id = v; }
    };

    std::vector<int32_t> gen;
    resp.finish_reason = QWEN3_ASR_FINISH_EOS;
    for (int32_t step = 0;; ++step) {
        if (step > 0) {
            if (!eval_token_graph(&gen.back(), nullptr, 1, kv_.n_past, logits))
                return fail("decode step graph failed");
        }
        int32_t best = 0;
        argmax(best);
        gen.push_back(best);

        // 停止条件：EOS 集合 / 生成上限 / KV 容量
        bool is_eos = false;
        for (int32_t e : model_.asr_hparams.eos_token_ids) if (best == e) { is_eos = true; break; }
        if (is_eos) break;
        if ((int32_t) gen.size() >= max_gen) { resp.finish_reason = QWEN3_ASR_FINISH_LENGTH; break; }
        if (kv_.n_past + 1 > kv_.n_ctx) { resp.finish_reason = QWEN3_ASR_FINISH_CONTEXT_FULL; break; }
    }

    // ---- 5. 文本与协议解析 ----
    const std::string raw = tok_.decode(gen, /*skip_special=*/false);
    std::string parsed_lang, text;
    parse_protocol(tok_, gen, parsed_lang, text);

    resp.ok = true;
    resp.text  = text;
    resp.language = !req.language.empty() ? req.language : parsed_lang;
    resp.n_gen = (int32_t) gen.size();
    if (req.want_debug) {
        resp.gen_ids = std::move(gen);
        resp.raw = raw;
    }
    return resp;
}
