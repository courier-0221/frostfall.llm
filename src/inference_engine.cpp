// frostfall v1.0 —— 通用推理接口实现（inference_engine.cpp）。
//
// M2 阶段：把 v0.3 main.cpp 的推理主链路迁进 InferenceEngine::Impl：
//   加载模型 + tokenizer + KV cache -> Qwen3PromptBuilder 拼 ChatML -> 编码 ->
//   prefill(整段) -> decode(逐 token, K/V 增量) -> 采样 -> 累积文本。
// 本阶段仅打通**非流式** Infer（返回完整 content）；流式帧拆分（thinking / tool_call）
// 与线程安全细化留待 M3 / M4。

#include "inference_engine.h"

#include "common.h"
#include "graph.h"
#include "kv_cache.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "qwen3_chat.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "log.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

namespace Frostfall {

namespace {

// 把 nlohmann JSON 解析成 EngineConfig；缺省字段保留 EngineConfig 的默认值。
bool parse_engine_config(const nlohmann::json& j, EngineConfig& cfg) {
    if (j.contains("model_path")) cfg.model_path = j.at("model_path").get<std::string>();
    if (j.contains("n_ctx"))      cfg.n_ctx      = j.at("n_ctx").get<int32_t>();
    if (j.contains("n_threads"))  cfg.n_threads  = j.at("n_threads").get<int32_t>();
    if (j.contains("max_tokens")) cfg.max_tokens = j.at("max_tokens").get<int32_t>();

    if (j.contains("sampling")) {
        const auto& s = j.at("sampling");
        SamplingParams& sp = cfg.sampling;
        if (s.contains("temp"))           sp.temp           = s.at("temp").get<float>();
        if (s.contains("top_k"))          sp.top_k          = s.at("top_k").get<int32_t>();
        if (s.contains("top_p"))          sp.top_p          = s.at("top_p").get<float>();
        if (s.contains("repeat_penalty")) sp.repeat_penalty = s.at("repeat_penalty").get<float>();
        if (s.contains("repeat_last_n"))  sp.repeat_last_n  = s.at("repeat_last_n").get<int32_t>();
        if (s.contains("seed"))           sp.seed           = s.at("seed").get<uint32_t>();
    }
    return !cfg.model_path.empty();
}

// SamplingParams(公共) -> sampler_params(内部)：字段一一对应。
sampler_params to_internal_sampling(const SamplingParams& s) {
    sampler_params p;
    p.temp           = s.temp;
    p.top_k          = s.top_k;
    p.top_p          = s.top_p;
    p.repeat_penalty = s.repeat_penalty;
    p.repeat_last_n  = s.repeat_last_n;
    p.seed           = s.seed;
    return p;
}

}  // namespace

struct InferenceEngine::Impl {
    EngineConfig config;
    bool         initialized = false;

    // 推理资源（M2 迁移自 main.cpp）
    qwen3_model     model;
    qwen3_tokenizer tokenizer;
    qwen3_kv_cache  kv;
    int32_t         n_ctx  = 0;             // 实际生效的上下文上限
    ggml_gallocr_t  allocr = nullptr;

    static constexpr int kMaxNodes = 8192;

    // eval_batch 复用缓冲
    std::vector<float>   mask_buf;
    std::vector<int32_t> pos_buf;

    // 线程安全
    std::mutex         infer_mutex;         // Init / Infer 串行互斥
    std::atomic<bool>  cancel_flag{false};
    mutable std::mutex stats_mutex;         // 保护 last_stats
    InferenceStats     last_stats;

    ~Impl() {
        if (allocr) ggml_gallocr_free(allocr);
    }

    bool is_eos(int32_t id) const {
        return id == tokenizer.eos_id
            || id == model.hparams.eos_token_id
            || id == model.hparams.eot_token_id;
    }

    // 前向一个 batch：把 [batch(n 个)] 写入 cache 的 [n_past, n_past+n)，
    // 对“最后一个位置”的 logits 采样返回下一个 token。prev 为采样前的完整历史（重复惩罚回看）。
    int32_t eval_batch(sampler& smpl, const int32_t* batch, int32_t n, int32_t n_past,
                       const std::vector<int32_t>& prev) {
        const int32_t n_vocab = model.hparams.n_vocab;
        const int32_t n_kv    = n_past + n;

        struct ggml_init_params gparams = {
            /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context* ctx = ggml_init(gparams);

        struct ggml_cgraph* gf = qwen3_build_graph(ctx, model, kv, n, n_past, kMaxNodes);
        ggml_gallocr_alloc_graph(allocr, gf);

        struct ggml_tensor* t_tokens = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_TOKENS);
        ggml_backend_tensor_set(t_tokens, batch, 0, n * sizeof(int32_t));

        pos_buf.resize(n);
        for (int32_t i = 0; i < n; ++i) pos_buf[i] = n_past + i;
        struct ggml_tensor* t_pos = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_POS);
        ggml_backend_tensor_set(t_pos, pos_buf.data(), 0, n * sizeof(int32_t));

        // 因果 mask：ne0=k(0..n_kv-1)，ne1=q(0..n-1)；query 绝对位置 n_past+q，可见 k<=n_past+q。
        mask_buf.assign((size_t) n * n_kv, 0.0f);
        for (int32_t q = 0; q < n; ++q) {
            for (int32_t k = n_past + q + 1; k < n_kv; ++k) {
                mask_buf[(size_t) q * n_kv + k] = -INFINITY;
            }
        }
        struct ggml_tensor* t_mask = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_MASK);
        ggml_backend_tensor_set(t_mask, mask_buf.data(), 0, mask_buf.size() * sizeof(float));

        ggml_backend_graph_compute(model.backend, gf);

        struct ggml_tensor* t_logits = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_LOGITS);
        std::vector<float> logits(n_vocab);
        ggml_backend_tensor_get(t_logits, logits.data(),
                (size_t)(n - 1) * n_vocab * sizeof(float), n_vocab * sizeof(float));

        ggml_free(ctx);
        kv.n_past = n_kv;
        return smpl.sample(logits.data(), n_vocab, prev);
    }
};

InferenceEngine::InferenceEngine() : impl_(std::make_unique<Impl>()) {}
InferenceEngine::~InferenceEngine() = default;

bool InferenceEngine::Init(const EngineConfig& config) {
    std::lock_guard<std::mutex> lock(impl_->infer_mutex);

    impl_->config      = config;
    impl_->initialized = false;
    if (config.model_path.empty()) {
        LOG(ERROR) << "EngineConfig.model_path is empty";
        return false;
    }

    if (!qwen3_model_load(config.model_path, impl_->model)) {
        LOG(ERROR) << "failed to load model from '" << config.model_path << "'";
        return false;
    }
    if (ggml_backend_is_cpu(impl_->model.backend)) {
        ggml_backend_cpu_set_n_threads(impl_->model.backend, config.n_threads);
    }

    if (!impl_->tokenizer.load(config.model_path)) {
        LOG(ERROR) << "failed to load tokenizer from '" << config.model_path << "'";
        return false;
    }

    int32_t n_ctx = config.n_ctx;
    if (n_ctx > impl_->model.hparams.n_ctx_train) n_ctx = impl_->model.hparams.n_ctx_train;
    impl_->n_ctx = n_ctx;

    if (!impl_->kv.init(impl_->model, n_ctx)) {
        LOG(ERROR) << "failed to init KV cache";
        return false;
    }

    impl_->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->model.backend));
    if (!impl_->allocr) {
        LOG(ERROR) << "failed to create graph allocator";
        return false;
    }

    impl_->initialized = true;
    LOG(INFO) << "engine initialized: n_ctx=" << n_ctx
              << ", n_threads=" << config.n_threads
              << ", max_tokens=" << config.max_tokens;
    return true;
}

bool InferenceEngine::InitFromJsonString(const std::string& json) {
    EngineConfig cfg;
    try {
        const auto j = nlohmann::json::parse(json);
        if (!parse_engine_config(j, cfg)) {
            LOG(ERROR) << "invalid engine config json (missing model_path?)";
            return false;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "failed to parse engine config json: " << e.what();
        return false;
    }
    return Init(cfg);
}

bool InferenceEngine::InitFromJsonFile(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        LOG(ERROR) << "failed to open engine config file '" << path << "'";
        return false;
    }
    std::stringstream ss;
    ss << fin.rdbuf();
    return InitFromJsonString(ss.str());
}

LlmResponse InferenceEngine::Infer(const LlmRequest& request,
                                   const StreamCallback& on_chunk) {
    std::lock_guard<std::mutex> lock(impl_->infer_mutex);
    impl_->cancel_flag.store(false);

    LlmResponse resp;
    resp.msg_id       = request.msg_id;
    resp.message.role = Role::ASSISTANT;
    resp.is_end       = true;

    if (!impl_->initialized) {
        LOG(ERROR) << "Infer called before successful Init";
        resp.finish_reason = FinishReason::ERROR;
        return resp;
    }

    timer t_total;
    timer t_stage;

    // ---- 1. 拼 ChatML 并编码 ----
    Qwen3PromptBuilder builder;
    const std::string prompt = builder.build(request.messages, request.enable_thinking);
    LOG(INFO) << "chatml prompt:\n" << prompt;
    std::vector<int32_t> ids = impl_->tokenizer.encode(prompt);
    if (ids.empty()) {
        LOG(ERROR) << "empty prompt token sequence";
        resp.finish_reason = FinishReason::ERROR;
        return resp;
    }
    const int32_t n_prompt = (int32_t) ids.size();

    // ---- 2. 生成上限 ----
    int32_t max_tokens = request.max_tokens > 0 ? request.max_tokens : impl_->config.max_tokens;
    if (max_tokens <= 0) max_tokens = 1;

    // ---- 3. 无状态：重置 KV cache ----
    impl_->kv.n_past = 0;

    // ---- 4. 采样器 ----
    sampler smpl;
    smpl.init(to_internal_sampling(impl_->config.sampling));

    InferenceStats stats;
    stats.prompt_token_count = n_prompt;

    // ---- 5. 流式拆分器与回调 ----
    // enable_thinking=true 时 prompt 末尾是 "<think>\n"，模型直接生成思考内容，
    // 因此 ThinkSplitter 初始即处于 thinking 态，直到看到 "</think>"。
    ThinkSplitter    think_splitter(request.enable_thinking);
    ToolCallSplitter tool_splitter;

    // 复用的流式帧对象；三类帧（thinking / content / tool_call）互斥下发。
    LlmResponse frame;
    frame.msg_id       = request.msg_id;
    frame.message.role = Role::ASSISTANT;
    const bool do_stream = request.stream && static_cast<bool>(on_chunk);

    auto emit_thinking = [&](const std::string& t) {
        if (!do_stream || t.empty()) return;
        frame.thinking = t;
        frame.message.content.clear();
        frame.message.tool_calls.clear();
        frame.is_end        = false;
        frame.finish_reason = FinishReason::NONE;
        on_chunk(frame);
    };
    auto emit_content = [&](const std::string& c) {
        if (!do_stream || c.empty()) return;
        frame.thinking.clear();
        frame.message.content = c;
        frame.message.tool_calls.clear();
        frame.is_end        = false;
        frame.finish_reason = FinishReason::NONE;
        on_chunk(frame);
    };
    auto emit_tool_call = [&](const ToolCall& tc) {
        if (!do_stream) return;
        frame.thinking.clear();
        frame.message.content.clear();
        frame.message.tool_calls.assign(1, tc);
        frame.is_end        = false;
        frame.finish_reason = FinishReason::NONE;
        on_chunk(frame);
    };

    // 处理一个新 token 的解码文本：ThinkSplitter -> ToolCallSplitter -> 累积 + 回调。
    // 用 skip_special=false，让 "</think>" / "<tool_call>" 等 USER_DEFINED token 以字面量出现，
    // 供 splitter 识别（eos 已在外层提前 break，不会走到这里）。
    auto process_token = [&](int32_t id) {
        const std::string piece = impl_->tokenizer.id_to_piece(id, /*skip_special=*/false);
        LOG(INFO) << "decoded token: id=" << id << ", piece=" << piece;

        std::string thinking_delta, after_think;
        think_splitter.feed(piece, thinking_delta, after_think);
        if (!thinking_delta.empty()) {
            resp.thinking += thinking_delta;
            emit_thinking(thinking_delta);
        }
        if (!after_think.empty()) {
            std::string           content_delta;
            std::vector<ToolCall> calls;
            tool_splitter.feed(after_think, content_delta, calls);
            if (!content_delta.empty()) {
                resp.message.content += content_delta;
                emit_content(content_delta);
            }
            for (auto& c : calls) {
                resp.message.tool_calls.push_back(c);
                emit_tool_call(c);
            }
        }
    };

    // ---- 6. prefill ----
    t_stage.reset();
    int32_t next_id = impl_->eval_batch(smpl, ids.data(), n_prompt, 0, ids);
    stats.ttft_ms = t_stage.elapsed_ms();
    const double t_prefill_ms = stats.ttft_ms;

    int32_t      n_decoded = 0;
    FinishReason finish    = FinishReason::LENGTH;

    // 首 token
    ids.push_back(next_id);
    ++n_decoded;
    if (impl_->is_eos(next_id)) {
        finish = FinishReason::STOP;
    } else {
        process_token(next_id);
    }

    // ---- 7. decode ----
    t_stage.reset();
    if (finish != FinishReason::STOP) {
        for (int32_t step = 1; step < max_tokens; ++step) {
            if (impl_->cancel_flag.load()) { finish = FinishReason::CANCELLED; break; }
            if (impl_->kv.n_past >= impl_->n_ctx) {
                LOG(WARNING) << "reached ctx-size (" << impl_->n_ctx << "), stopping";
                finish = FinishReason::LENGTH;
                break;
            }

            next_id = impl_->eval_batch(smpl, &next_id, 1, impl_->kv.n_past, ids);
            ids.push_back(next_id);
            ++n_decoded;

            if (impl_->is_eos(next_id)) { finish = FinishReason::STOP; break; }

            process_token(next_id);
        }
    }
    const double t_decode_ms = t_stage.elapsed_ms();

    // ---- 8. flush 拆分器残留 ----
    {
        std::string thinking_delta;
        think_splitter.flush(thinking_delta);
        if (!thinking_delta.empty()) {
            resp.thinking += thinking_delta;
            emit_thinking(thinking_delta);
        }
    }
    {
        std::string           content_delta;
        std::vector<ToolCall> calls;
        tool_splitter.flush(content_delta, calls);
        if (!content_delta.empty()) {
            resp.message.content += content_delta;
            emit_content(content_delta);
        }
        for (auto& c : calls) {
            resp.message.tool_calls.push_back(c);
            emit_tool_call(c);
        }
    }

    // 本轮产出过工具调用且是自然结束 -> TOOL_CALLS。
    if (finish == FinishReason::STOP && !resp.message.tool_calls.empty()) {
        finish = FinishReason::TOOL_CALLS;
    }

    // ---- 9. 统计 ----
    stats.gen_token_count = n_decoded;
    stats.prompt_tokens_per_sec = t_prefill_ms > 0 ? n_prompt * 1000.0 / t_prefill_ms : 0.0;
    // decode 吞吐：不含首 token（首 token 计入 prefill/TTFT 段）。
    const int32_t n_dec_only = n_decoded > 1 ? n_decoded - 1 : 0;
    stats.gen_tokens_per_sec = (n_dec_only > 0 && t_decode_ms > 0)
                                   ? n_dec_only * 1000.0 / t_decode_ms : 0.0;

    {
        std::lock_guard<std::mutex> slock(impl_->stats_mutex);
        impl_->last_stats = stats;
    }

    LOG(INFO) << "infer: prompt=" << n_prompt << " tok, gen=" << n_decoded
              << " tok, ttft=" << stats.ttft_ms << " ms, decode=" << t_decode_ms
              << " ms, total=" << t_total.elapsed_ms() << " ms";

    // ---- 10. 组装最终响应 / 终止帧 ----
    resp.finish_reason = finish;
    if (do_stream) {
        LlmResponse end_frame;
        end_frame.msg_id        = request.msg_id;
        end_frame.message.role  = Role::ASSISTANT;
        end_frame.is_end        = true;
        end_frame.finish_reason = finish;
        on_chunk(end_frame);
    }
    return resp;
}

void InferenceEngine::Cancel() {
    impl_->cancel_flag.store(true);
}

InferenceStats InferenceEngine::GetLastStats() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return impl_->last_stats;
}

int InferenceEngine::MaxTokens() const {
    return impl_->config.max_tokens;
}

}  // namespace Frostfall
