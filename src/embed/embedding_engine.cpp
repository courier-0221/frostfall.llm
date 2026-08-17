#include "embedding_engine.h"

#include "common.h"
#include "embed_graph.h"
#include "log.h"
#include "model.h"
#include "pooling.h"
#include "qwen3_embed_prompt.h"
#include "tokenizer.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Frostfall {

namespace {

bool parse_engine_config(const nlohmann::json& j, EmbeddingEngine::EngineConfig& cfg) {
    if (j.contains("model_path"))   cfg.model_path   = j.at("model_path").get<std::string>();
    if (j.contains("n_ctx"))        cfg.n_ctx        = j.at("n_ctx").get<int>();
    if (j.contains("n_threads"))    cfg.n_threads    = j.at("n_threads").get<int>();
    if (j.contains("max_batch"))    cfg.max_batch    = j.at("max_batch").get<int>();
    if (j.contains("target_dim"))   cfg.target_dim   = j.at("target_dim").get<int>();
    if (j.contains("pad_token_id")) cfg.pad_token_id = j.at("pad_token_id").get<int>();
    if (j.contains("default_task")) cfg.default_task = j.at("default_task").get<std::string>();
    return !cfg.model_path.empty();
}

int resolve_target_dim(int requested, int fallback, int n_embd) {
    int dim = requested > 0 ? requested : fallback;
    if (dim <= 0 || dim >= n_embd) return n_embd;
    if (dim < 32) {
        throw std::runtime_error("target_dim must be 0/full or in [32, n_embd]");
    }
    return dim;
}

}  // namespace

struct EmbeddingEngine::Impl {
    EngineConfig config;
    bool initialized = false;

    std::unique_ptr<qwen3_model> model;
    qwen3_tokenizer tokenizer;
    ggml_gallocr_t allocr = nullptr;
    int n_ctx = 0;
    double load_ms = 0.0;

    static constexpr int kMaxNodes = 8192;

    std::mutex infer_mutex;
    mutable std::mutex stats_mutex;
    EmbedStats last_stats;

    ~Impl() {
        if (allocr) ggml_gallocr_free(allocr);
    }

    int eos_id() const {
        if (model && model->hparams.eos_token_id >= 0) return model->hparams.eos_token_id;
        if (tokenizer.eos_id >= 0) return tokenizer.eos_id;
        return 151643;
    }

    int pad_id() const {
        if (config.pad_token_id >= 0) return config.pad_token_id;
        if (tokenizer.pad_id >= 0) return tokenizer.pad_id;
        return eos_id();
    }
};

EmbeddingEngine::EmbeddingEngine() : impl_(std::make_unique<Impl>()) {}
EmbeddingEngine::~EmbeddingEngine() = default;

bool EmbeddingEngine::Init(const EngineConfig& cfg) {
    std::lock_guard<std::mutex> lock(impl_->infer_mutex);

    if (cfg.model_path.empty()) {
        LOG(ERROR) << "EmbeddingEngine config model_path is empty";
        return false;
    }
    if (cfg.n_ctx <= 0 || cfg.n_threads <= 0 || cfg.max_batch <= 0) {
        LOG(ERROR) << "EmbeddingEngine config n_ctx/n_threads/max_batch must be positive";
        return false;
    }

    timer t_load;
    impl_->initialized = false;
    impl_->config = cfg;
    impl_->tokenizer = qwen3_tokenizer{};
    if (impl_->allocr) {
        ggml_gallocr_free(impl_->allocr);
        impl_->allocr = nullptr;
    }
    impl_->model = std::make_unique<qwen3_model>();

    if (!qwen3_model_load(cfg.model_path, *impl_->model)) {
        LOG(ERROR) << "failed to load embedding model from '" << cfg.model_path << "'";
        impl_->model.reset();
        return false;
    }
    if (ggml_backend_is_cpu(impl_->model->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->model->backend, cfg.n_threads);
    }
    if (!impl_->tokenizer.load(cfg.model_path)) {
        LOG(ERROR) << "failed to load tokenizer from '" << cfg.model_path << "'";
        impl_->model.reset();
        return false;
    }

    impl_->n_ctx = std::min(cfg.n_ctx, impl_->model->hparams.n_ctx_train);
    impl_->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->model->backend));
    if (!impl_->allocr) {
        LOG(ERROR) << "failed to create embedding graph allocator";
        impl_->model.reset();
        return false;
    }

    impl_->load_ms = t_load.elapsed_ms();
    {
        std::lock_guard<std::mutex> slock(impl_->stats_mutex);
        impl_->last_stats = EmbedStats{};
        impl_->last_stats.load_ms = impl_->load_ms;
    }

    impl_->initialized = true;
    LOG(INFO) << "embedding engine initialized: n_ctx=" << impl_->n_ctx
              << ", n_threads=" << cfg.n_threads
              << ", max_batch=" << cfg.max_batch
              << ", target_dim=" << cfg.target_dim;
    return true;
}

bool EmbeddingEngine::InitFromJsonString(const std::string& json) {
    EngineConfig cfg;
    try {
        const auto j = nlohmann::json::parse(json);
        if (!parse_engine_config(j, cfg)) {
            LOG(ERROR) << "invalid embedding config json (missing model_path?)";
            return false;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "failed to parse embedding config json: " << e.what();
        return false;
    }
    return Init(cfg);
}

bool EmbeddingEngine::InitFromJsonFile(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        LOG(ERROR) << "failed to open embedding config file '" << path << "'";
        return false;
    }
    std::stringstream ss;
    ss << fin.rdbuf();
    return InitFromJsonString(ss.str());
}

EmbedResponse EmbeddingEngine::Embed(const EmbedRequest& req) {
    std::lock_guard<std::mutex> lock(impl_->infer_mutex);

    EmbedResponse resp;
    resp.stats.load_ms = impl_->load_ms;

    if (!impl_->initialized || !impl_->model) {
        resp.error = "EmbeddingEngine is not initialized";
        return resp;
    }
    if (req.texts.empty()) {
        resp.error = "EmbedRequest.texts is empty";
        return resp;
    }
    if ((int) req.texts.size() > impl_->config.max_batch) {
        resp.error = "batch size exceeds EngineConfig.max_batch";
        return resp;
    }

    try {
        const int n_embd = impl_->model->hparams.n_embd;
        const int out_dim = resolve_target_dim(req.target_dim, impl_->config.target_dim, n_embd);
        resp.dim = out_dim;

        timer t_stage;
        std::vector<std::vector<int32_t>> seqs;
        seqs.reserve(req.texts.size());
        int max_len = 0;
        const std::string& task = req.task.empty() ? impl_->config.default_task : req.task;
        const int eos = impl_->eos_id();

        for (const std::string& text : req.texts) {
            const std::string input = req.is_query ? qwen3_embed_build_query_text(text, task) : text;
            std::vector<int32_t> ids = impl_->tokenizer.encode(input);
            ids.push_back(eos);
            if (ids.empty()) {
                resp.error = "empty token sequence after tokenization";
                return resp;
            }
            max_len = std::max(max_len, (int) ids.size());
            seqs.push_back(std::move(ids));
        }
        resp.stats.tokenize_ms = t_stage.elapsed_ms();
        resp.stats.n_tokens_max = max_len;
        resp.stats.n_texts = (int) seqs.size();

        if (max_len > impl_->n_ctx) {
            resp.error = "tokenized input exceeds EngineConfig.n_ctx";
            return resp;
        }

        const int batch = (int) seqs.size();
        const int total_tokens = max_len * batch;
        const int pad = impl_->pad_id();

        std::vector<int32_t> tokens((size_t) total_tokens, pad);
        std::vector<int32_t> positions((size_t) total_tokens, 0);
        std::vector<int> pad_prefix((size_t) batch, 0);

        for (int b = 0; b < batch; ++b) {
            const int len = (int) seqs[b].size();
            const int left = max_len - len;
            pad_prefix[b] = left;
            for (int i = 0; i < len; ++i) {
                const int flat = b * max_len + left + i;
                tokens[flat] = seqs[b][i];
                positions[flat] = i;
            }
        }

        const float neg_inf = -std::numeric_limits<float>::infinity();
        std::vector<float> mask((size_t) total_tokens * total_tokens, neg_inf);
        for (int b = 0; b < batch; ++b) {
            const int left = pad_prefix[b];
            for (int q = 0; q < max_len; ++q) {
                const int q_flat = b * max_len + q;
                for (int k = 0; k <= q; ++k) {
                    const bool q_is_pad = q < left;
                    const bool k_is_pad = k < left;
                    if (!q_is_pad && k_is_pad) continue;
                    const int k_flat = b * max_len + k;
                    mask[(size_t) q_flat * total_tokens + k_flat] = 0.0f;
                }
            }
        }

        t_stage.reset();
        struct ggml_init_params gparams = {
            /*.mem_size   =*/ ggml_tensor_overhead() * Impl::kMaxNodes
                              + ggml_graph_overhead_custom(Impl::kMaxNodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context* ctx = ggml_init(gparams);
        if (!ctx) {
            resp.error = "failed to create ggml context";
            return resp;
        }

        struct ggml_cgraph* gf = qwen3_embed_build_graph(ctx, *impl_->model, total_tokens, Impl::kMaxNodes);
        ggml_gallocr_alloc_graph(impl_->allocr, gf);

        struct ggml_tensor* t_tokens = ggml_graph_get_tensor(gf, QWEN3_EMBED_TENSOR_NAME_TOKENS);
        ggml_backend_tensor_set(t_tokens, tokens.data(), 0, tokens.size() * sizeof(int32_t));

        struct ggml_tensor* t_pos = ggml_graph_get_tensor(gf, QWEN3_EMBED_TENSOR_NAME_POS);
        ggml_backend_tensor_set(t_pos, positions.data(), 0, positions.size() * sizeof(int32_t));

        struct ggml_tensor* t_mask = ggml_graph_get_tensor(gf, QWEN3_EMBED_TENSOR_NAME_MASK);
        ggml_backend_tensor_set(t_mask, mask.data(), 0, mask.size() * sizeof(float));

        ggml_backend_graph_compute(impl_->model->backend, gf);
        resp.stats.compute_ms = t_stage.elapsed_ms();

        t_stage.reset();
        struct ggml_tensor* t_hidden = ggml_graph_get_tensor(gf, QWEN3_EMBED_TENSOR_NAME_HIDDEN);
        std::vector<float> last((size_t) n_embd);
        resp.embeddings.reserve(batch);
        for (int b = 0; b < batch; ++b) {
            const int last_index = b * max_len + (max_len - 1);
            const size_t offset = (size_t) last_index * n_embd * sizeof(float);
            ggml_backend_tensor_get(t_hidden, last.data(), offset, last.size() * sizeof(float));
            resp.embeddings.push_back(qwen3_embed_pool_and_normalize(last.data(), n_embd, 1, out_dim));
        }
        resp.stats.pool_ms = t_stage.elapsed_ms();

        ggml_free(ctx);

        {
            std::lock_guard<std::mutex> slock(impl_->stats_mutex);
            impl_->last_stats = resp.stats;
        }
        LOG(INFO) << "embed: batch=" << batch << ", max_tokens=" << max_len
                  << ", dim=" << out_dim << ", tokenize=" << resp.stats.tokenize_ms
                  << " ms, compute=" << resp.stats.compute_ms
                  << " ms, pool=" << resp.stats.pool_ms << " ms";
        return resp;
    } catch (const std::exception& e) {
        resp.error = e.what();
        return resp;
    }
}

std::vector<float> EmbeddingEngine::Embed(const std::string& text,
                                          bool is_query,
                                          const std::string& task) {
    EmbedRequest req;
    req.texts.push_back(text);
    req.is_query = is_query;
    req.task = task;
    EmbedResponse resp = Embed(req);
    if (!resp.error.empty() || resp.embeddings.empty()) return {};
    return resp.embeddings.front();
}

EmbedStats EmbeddingEngine::GetLastStats() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return impl_->last_stats;
}

}  // namespace Frostfall
