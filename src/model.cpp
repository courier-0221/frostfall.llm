#include "model.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "log.h"

#include <algorithm>
#include <iomanip>
#include <vector>

qwen3_model::~qwen3_model() {
    if (buffer)   ggml_backend_buffer_free(buffer);
    if (backend)  ggml_backend_free(backend);
    if (ctx_data) ggml_free(ctx_data);
}

namespace {

// 读取 GGUF 元数据里的整型 kv，兼容常见的几种整型存储方式；key 不存在时返回默认值。
int32_t gguf_get_i32_any(const gguf_context * ctx, const char * key, int32_t def) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        return def;
    }
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_UINT32:  return (int32_t) gguf_get_val_u32(ctx, kid);
        case GGUF_TYPE_INT32:   return           gguf_get_val_i32(ctx, kid);
        case GGUF_TYPE_UINT64:  return (int32_t) gguf_get_val_u64(ctx, kid);
        case GGUF_TYPE_INT64:   return (int32_t) gguf_get_val_i64(ctx, kid);
        default:                return def;
    }
}

float gguf_get_f32_any(const gguf_context * ctx, const char * key, float def) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        return def;
    }
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, kid);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(ctx, kid);
        default:                return def;
    }
}

// 把 GGUF 文件里的原始权重字节，按张量名拷贝进已经分配好的 backend buffer。
// 写法照抄 ggml 官方 examples/mnist 的 load_from_gguf()（见 doc/design.md 里的参考）。
bool qwen3_load_tensor_data(const std::string & fname, ggml_context * ctx_data, gguf_context * ctx_gguf) {
    FILE * f = ggml_fopen(fname.c_str(), "rb");
    if (!f) {
        LOG(ERROR) << __func__ << ": failed to reopen '" << fname << "' for reading weights";
        return false;
    }

    const size_t buf_size = 16 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);

        ggml_tensor * tensor = ggml_get_tensor(ctx_data, name);
        if (!tensor) {
            continue; // GGUF 里有但我们没引用到的张量（理论上不应该出现），忽略
        }

        const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        if (fseek(f, (long) offs, SEEK_SET) != 0) {
            fclose(f);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t chunk = std::min(buf_size, nbytes - pos);
            if (fread(buf.data(), 1, chunk, f) != chunk) {
                fclose(f);
                return false;
            }
            ggml_backend_tensor_set(tensor, buf.data(), pos, chunk);
        }
    }

    fclose(f);
    return true;
}

} // namespace

bool qwen3_model_load(const std::string & fname, qwen3_model & model) {
    // no_alloc=true：只建立张量的元数据（名字/形状/类型），不分配数据内存；
    // ctx=&model.ctx_data：gguf_init_from_file 会自动创建这个 ggml_context 并把 GGUF 里
    // 记录的每个张量都建好（这样我们后面直接按名字 ggml_get_tensor() 取指针即可，不用手写 new_tensor）。
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &model.ctx_data,
    };

    gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        LOG(ERROR) << __func__ << ": gguf_init_from_file() failed for '" << fname << "'";
        return false;
    }

    const int64_t kid_arch = gguf_find_key(ctx_gguf, "general.architecture");
    const std::string arch = kid_arch >= 0 ? gguf_get_val_str(ctx_gguf, kid_arch) : "";
    if (arch != "qwen3") {
        LOG(ERROR) << __func__ << ": unsupported architecture '" << arch << "' (this framework only supports 'qwen3')";
        gguf_free(ctx_gguf);
        return false;
    }

    qwen3_hparams & hp = model.hparams;
    hp.n_embd         = gguf_get_i32_any(ctx_gguf, "qwen3.embedding_length", 0);
    hp.n_layer        = gguf_get_i32_any(ctx_gguf, "qwen3.block_count", 0);
    hp.n_head         = gguf_get_i32_any(ctx_gguf, "qwen3.attention.head_count", 0);
    hp.n_head_kv      = gguf_get_i32_any(ctx_gguf, "qwen3.attention.head_count_kv", hp.n_head);
    hp.n_embd_head    = gguf_get_i32_any(ctx_gguf, "qwen3.attention.key_length", 0);
    hp.n_ff           = gguf_get_i32_any(ctx_gguf, "qwen3.feed_forward_length", 0);
    hp.n_ctx_train    = gguf_get_i32_any(ctx_gguf, "qwen3.context_length", 4096);
    hp.rms_norm_eps   = gguf_get_f32_any(ctx_gguf, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    hp.rope_freq_base = gguf_get_f32_any(ctx_gguf, "qwen3.rope.freq_base", 1000000.0f);
    hp.eos_token_id   = gguf_get_i32_any(ctx_gguf, "tokenizer.ggml.eos_token_id", -1);
    hp.eot_token_id   = gguf_get_i32_any(ctx_gguf, "tokenizer.ggml.eot_token_id", 151645); // Qwen3 <|im_end|>
    hp.bos_token_id   = gguf_get_i32_any(ctx_gguf, "tokenizer.ggml.bos_token_id", -1);

    // 按名字取权重张量指针（张量本身已经由 gguf_init_from_file 建好在 model.ctx_data 里了）。
    auto get_tensor = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(model.ctx_data, name.c_str());
        if (!t) {
            LOG(ERROR) << __func__ << ": missing tensor '" << name << "'";
        }
        return t;
    };

    model.tok_embd = get_tensor("token_embd.weight");
    if (!model.tok_embd) {
        gguf_free(ctx_gguf);
        return false;
    }
    // vocab size 从 token embedding 的形状读，比元数据里的 key 更可靠。
    hp.n_vocab = (int32_t) model.tok_embd->ne[1];

    if (hp.n_embd <= 0 || hp.n_layer <= 0 || hp.n_head <= 0 || hp.n_embd_head <= 0 || hp.n_ff <= 0) {
        LOG(ERROR) << __func__ << ": incomplete hparams read from GGUF metadata";
        gguf_free(ctx_gguf);
        return false;
    }

    LOG(INFO) << __func__
              << ": n_vocab=" << hp.n_vocab
              << " n_embd=" << hp.n_embd
              << " n_layer=" << hp.n_layer
              << " n_head=" << hp.n_head
              << " n_head_kv=" << hp.n_head_kv
              << " n_embd_head=" << hp.n_embd_head
              << " n_ff=" << hp.n_ff
              << " eps=" << hp.rms_norm_eps
              << " rope_base=" << hp.rope_freq_base
              << " eos=" << hp.eos_token_id;

    model.output_norm = get_tensor("output_norm.weight");
    model.output      = ggml_get_tensor(model.ctx_data, "output.weight"); // 可能不存在（tied embedding）
    if (!model.output) {
        model.output = model.tok_embd;
        LOG(INFO) << __func__ << ": 'output.weight' not found, reusing token_embd (tied embedding)";
    }

    if (!model.output_norm) {
        gguf_free(ctx_gguf);
        return false;
    }

    model.layers.resize(hp.n_layer);
    bool ok = true;
    for (int32_t il = 0; il < hp.n_layer; ++il) {
        qwen3_layer & layer = model.layers[il];
        const std::string p = "blk." + std::to_string(il) + ".";

        layer.attn_norm   = get_tensor(p + "attn_norm.weight");
        layer.wq          = get_tensor(p + "attn_q.weight");
        layer.wk          = get_tensor(p + "attn_k.weight");
        layer.wv          = get_tensor(p + "attn_v.weight");
        layer.wo          = get_tensor(p + "attn_output.weight");
        layer.attn_q_norm = get_tensor(p + "attn_q_norm.weight");
        layer.attn_k_norm = get_tensor(p + "attn_k_norm.weight");
        layer.ffn_norm    = get_tensor(p + "ffn_norm.weight");
        layer.ffn_gate    = get_tensor(p + "ffn_gate.weight");
        layer.ffn_up      = get_tensor(p + "ffn_up.weight");
        layer.ffn_down    = get_tensor(p + "ffn_down.weight");

        ok = ok && layer.attn_norm && layer.wq && layer.wk && layer.wv && layer.wo &&
             layer.attn_q_norm && layer.attn_k_norm && layer.ffn_norm &&
             layer.ffn_gate && layer.ffn_up && layer.ffn_down;
    }

    if (!ok) {
        LOG(ERROR) << __func__ << ": model is missing required tensors (see messages above)";
        gguf_free(ctx_gguf);
        return false;
    }

    // 初始化 CPU backend，把 ctx_data 里所有张量实际分配到 backend buffer 上。
    model.backend = ggml_backend_cpu_init();
    if (!model.backend) {
        LOG(ERROR) << __func__ << ": failed to init CPU backend";
        gguf_free(ctx_gguf);
        return false;
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx_data, model.backend);
    if (!model.buffer) {
        LOG(ERROR) << __func__ << ": failed to allocate backend buffer for weights";
        gguf_free(ctx_gguf);
        return false;
    }

    if (!qwen3_load_tensor_data(fname, model.ctx_data, ctx_gguf)) {
        LOG(ERROR) << __func__ << ": failed to read weight data from '" << fname << "'";
        gguf_free(ctx_gguf);
        return false;
    }

    LOG(INFO) << __func__ << ": weights buffer size = "
              << std::fixed << std::setprecision(2)
              << ggml_backend_buffer_get_size(model.buffer) / 1024.0 / 1024.0 << " MB";

    gguf_free(ctx_gguf);
    return true;
}
