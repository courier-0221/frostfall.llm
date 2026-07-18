#include "kv_cache.h"

#include "ggml-alloc.h"

#include "log.h"

#include <iomanip>

qwen3_kv_cache::~qwen3_kv_cache() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx)    ggml_free(ctx);
}

bool qwen3_kv_cache::init(const qwen3_model & model, int32_t n_ctx_) {
    const qwen3_hparams & hp = model.hparams;
    n_ctx  = n_ctx_;
    n_past = 0;

    const int64_t n_embd_kv_all = hp.n_embd_kv_all(); // head_dim * n_head_kv = 128 * 8
    const int64_t n_layer       = hp.n_layer;

    // 只放张量元数据，实际数据由后面 ggml_backend_alloc_ctx_tensors 统一分配。
    struct ggml_init_params params = {
        /*.mem_size   =*/ (size_t) (2 * n_layer + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    if (!ctx) {
        LOG(ERROR) << __func__ << ": ggml_init failed for kv cache";
        return false;
    }

    k.resize(n_layer);
    v.resize(n_layer);
    for (int64_t il = 0; il < n_layer; ++il) {
        k[il] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd_kv_all * n_ctx);
        v[il] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd_kv_all * n_ctx);
        ggml_set_name(k[il], ("cache_k_" + std::to_string(il)).c_str());
        ggml_set_name(v[il], ("cache_v_" + std::to_string(il)).c_str());
    }

    buffer = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    if (!buffer) {
        LOG(ERROR) << __func__ << ": failed to allocate kv cache buffer";
        return false;
    }

    LOG(INFO) << __func__ << ": kv cache allocated, n_ctx=" << n_ctx
              << ", size = " << std::fixed << std::setprecision(2)
              << size_bytes() / 1024.0 / 1024.0 << " MB";
    return true;
}

size_t qwen3_kv_cache::size_bytes() const {
    return buffer ? ggml_backend_buffer_get_size(buffer) : 0;
}
