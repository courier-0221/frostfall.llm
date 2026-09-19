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
    // 普通文本模型 = "qwen3"；ASR 单文件模型 = "qwen3-asr"（音频塔张量一并加载，见下）。
    if (arch != "qwen3" && arch != "qwen3-asr") {
        LOG(ERROR) << __func__ << ": unsupported architecture '" << arch
                   << "' (this framework supports 'qwen3' and 'qwen3-asr')";
        gguf_free(ctx_gguf);
        return false;
    }
    model.is_asr = (arch == "qwen3-asr");
    // 超参键名前缀随架构变化（"qwen3.xxx" / "qwen3-asr.xxx"）
    const std::string P = arch + ".";

    qwen3_hparams & hp = model.hparams;
    hp.n_embd         = gguf_get_i32_any(ctx_gguf, (P + "embedding_length").c_str(), 0);
    hp.n_layer        = gguf_get_i32_any(ctx_gguf, (P + "block_count").c_str(), 0);
    hp.n_head         = gguf_get_i32_any(ctx_gguf, (P + "attention.head_count").c_str(), 0);
    hp.n_head_kv      = gguf_get_i32_any(ctx_gguf, (P + "attention.head_count_kv").c_str(), hp.n_head);
    hp.n_embd_head    = gguf_get_i32_any(ctx_gguf, (P + "attention.key_length").c_str(), 0);
    hp.n_ff           = gguf_get_i32_any(ctx_gguf, (P + "feed_forward_length").c_str(), 0);
    hp.n_ctx_train    = gguf_get_i32_any(ctx_gguf, (P + "context_length").c_str(), 4096);
    hp.rms_norm_eps   = gguf_get_f32_any(ctx_gguf, (P + "attention.layer_norm_rms_epsilon").c_str(), 1e-6f);
    hp.rope_freq_base = gguf_get_f32_any(ctx_gguf, (P + "rope.freq_base").c_str(), 1000000.0f);
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

    // ==================== ASR（arch = qwen3-asr）====================
    // v0.1：音频塔张量全部加载到内存并校验存在性与形状，但不参与计算图。
    // any-asr 校验失败视为加载失败，避免静默丢弃权重。
    if (model.is_asr) {
        qwen3_asr_hparams & ah = model.asr_hparams;
        ah.n_audio_layer  = gguf_get_i32_any(ctx_gguf, "asr.audio.encoder_layers", 0);
        ah.d_model        = gguf_get_i32_any(ctx_gguf, "asr.audio.d_model", 0);
        ah.n_head         = gguf_get_i32_any(ctx_gguf, "asr.audio.attention.head_count", 0);
        ah.n_ff           = gguf_get_i32_any(ctx_gguf, "asr.audio.encoder_ffn_dim", 0);
        ah.output_dim     = gguf_get_i32_any(ctx_gguf, "asr.audio.output_dim", 0);
        ah.num_mel_bins   = gguf_get_i32_any(ctx_gguf, "asr.audio.num_mel_bins", 0);
        ah.n_window       = gguf_get_i32_any(ctx_gguf, "asr.audio.n_window", 0);
        ah.n_window_infer = gguf_get_i32_any(ctx_gguf, "asr.audio.n_window_infer", 0);
        ah.conv_chunksize = gguf_get_i32_any(ctx_gguf, "asr.audio.conv_chunksize", 0);
        ah.max_source_positions = gguf_get_i32_any(ctx_gguf, "asr.audio.max_source_positions", 0);
        ah.conv_kernel    = gguf_get_i32_any(ctx_gguf, "asr.audio.conv.kernel", 3);
        ah.conv_stride    = gguf_get_i32_any(ctx_gguf, "asr.audio.conv.stride", 2);
        ah.conv_padding   = gguf_get_i32_any(ctx_gguf, "asr.audio.conv.padding", 1);
        ah.layer_norm_eps = gguf_get_f32_any(ctx_gguf, "asr.audio.layer_norm_eps", 1e-5f);
        ah.audio_start_token_id = gguf_get_i32_any(ctx_gguf, "asr.audio_start_token_id", -1);
        ah.audio_end_token_id   = gguf_get_i32_any(ctx_gguf, "asr.audio_end_token_id", -1);
        ah.audio_token_id       = gguf_get_i32_any(ctx_gguf, "asr.audio_token_id", -1);
        {
            const int64_t kid = gguf_find_key(ctx_gguf, "asr.eos_token_ids");
            if (kid >= 0 && gguf_get_kv_type(ctx_gguf, kid) == GGUF_TYPE_ARRAY) {
                const int64_t n = gguf_get_arr_n(ctx_gguf, kid);
                const void * arr = gguf_get_arr_data(ctx_gguf, kid);
                for (int64_t i = 0; i < n; ++i) {
                    ah.eos_token_ids.push_back(((const int32_t *) arr)[i]);
                }
            }
        }
        if (ah.n_audio_layer <= 0 || ah.d_model <= 0 || ah.n_head <= 0 || ah.n_ff <= 0 ||
            ah.output_dim != hp.n_embd || ah.num_mel_bins <= 0 || ah.audio_token_id < 0) {
            LOG(ERROR) << __func__ << ": incomplete/invalid asr.* metadata"
                       << " (audio.output_dim must match text embedding_length)";
            gguf_free(ctx_gguf);
            return false;
        }

        // 张量读取 + 形状校验（im2col filter 布局：PyTorch [out,in,KH,KW] -> ne={KW,KH,in,out}）
        auto get_a = [&](const char * name, std::initializer_list<int64_t> ne) -> ggml_tensor * {
            ggml_tensor * t = get_tensor(name);
            if (!t) return nullptr;
            bool shape_ok = ggml_n_dims(t) == (int) ne.size();
            if (shape_ok) {
                const int64_t want[] = {ne.begin()[0], ne.size() > 1 ? ne.begin()[1] : 1,
                                        ne.size() > 2 ? ne.begin()[2] : 1,
                                        ne.size() > 3 ? ne.begin()[3] : 1};
                for (size_t d = 0; d < ne.size(); ++d) shape_ok = shape_ok && t->ne[d] == want[d];
            }
            if (!shape_ok) {
                std::string want_str, got_str;
                for (size_t d = 0; d < ne.size(); ++d) {
                    if (d) want_str += "x";
                    want_str += std::to_string(ne.begin()[d]);
                }
                for (int d = 0; d < ggml_n_dims(t); ++d) {
                    if (d) got_str += "x";
                    got_str += std::to_string(t->ne[d]);
                }
                LOG(ERROR) << __func__ << ": tensor '" << name << "' has unexpected shape, want "
                           << want_str << " got " << got_str;
                return nullptr;
            }
            return t;
        };

        // CNN 输出通道数不在 config 中（480），从 conv1.weight 的 ne[3] 推导并自洽校验
        {
            struct ggml_tensor * conv1_w_raw = get_tensor("asr.audio.conv1.weight");
            if (!conv1_w_raw || ggml_n_dims(conv1_w_raw) != 4) {
                LOG(ERROR) << __func__ << ": missing/mis-shaped asr.audio.conv1.weight";
                gguf_free(ctx_gguf);
                return false;
            }
            ah.conv_channels = (int32_t) conv1_w_raw->ne[3];
        }
        const int64_t conv_c = ah.conv_channels;
        // 频率轴经 3 层 k=3/s=2/p=1 卷积：out = (in+2p-k)/s+1，128 -> 64 -> 32 -> 16
        auto conv_out_len = [&](int64_t in) {
            return (in + 2 * (int64_t) ah.conv_padding - (int64_t) ah.conv_kernel)
                       / (int64_t) ah.conv_stride + 1;
        };
        const int64_t conv_freq_out = conv_out_len(conv_out_len(conv_out_len(ah.num_mel_bins)));

        model.a_conv1_w = get_a("asr.audio.conv1.weight", {3, 3, 1, conv_c});
        model.a_conv1_b = get_a("asr.audio.conv1.bias",   {conv_c});
        model.a_conv2_w = get_a("asr.audio.conv2.weight", {3, 3, conv_c, conv_c});
        model.a_conv2_b = get_a("asr.audio.conv2.bias",   {conv_c});
        model.a_conv3_w = get_a("asr.audio.conv3.weight", {3, 3, conv_c, conv_c});
        model.a_conv3_b = get_a("asr.audio.conv3.bias",   {conv_c});
        const int64_t conv_out_in = conv_c * conv_freq_out; // 480*16=7680
        model.a_conv_out_w = get_a("asr.audio.conv_out.weight", {conv_out_in, ah.d_model});
        model.a_ln_post_w  = get_a("asr.audio.ln_post.weight", {ah.d_model});
        model.a_ln_post_b  = get_a("asr.audio.ln_post.bias",   {ah.d_model});
        model.a_proj1_w    = get_a("asr.audio.proj1.weight", {ah.d_model, ah.d_model});
        model.a_proj1_b    = get_a("asr.audio.proj1.bias",   {ah.d_model});
        model.a_proj2_w    = get_a("asr.audio.proj2.weight", {ah.d_model, ah.output_dim}); // ne={in,out}
        model.a_proj2_b    = get_a("asr.audio.proj2.bias",   {ah.output_dim});

        model.asr_audio_layers.resize(ah.n_audio_layer);
        for (int32_t il = 0; il < ah.n_audio_layer && ok; ++il) {
            qwen3_asr_audio_layer & L = model.asr_audio_layers[il];
            const std::string p = "asr.audio.blk." + std::to_string(il) + ".";

            L.attn_q_w = get_a((p + "attn_q.weight").c_str(),  {ah.d_model, ah.d_model});
            L.attn_q_b = get_a((p + "attn_q.bias").c_str(),    {ah.d_model});
            L.attn_k_w = get_a((p + "attn_k.weight").c_str(),  {ah.d_model, ah.d_model});
            L.attn_k_b = get_a((p + "attn_k.bias").c_str(),    {ah.d_model});
            L.attn_v_w = get_a((p + "attn_v.weight").c_str(),  {ah.d_model, ah.d_model});
            L.attn_v_b = get_a((p + "attn_v.bias").c_str(),    {ah.d_model});
            L.attn_out_w = get_a((p + "attn_out.weight").c_str(), {ah.d_model, ah.d_model});
            L.attn_out_b = get_a((p + "attn_out.bias").c_str(),   {ah.d_model});
            L.attn_norm_w = get_a((p + "attn_norm.weight").c_str(), {ah.d_model});
            L.attn_norm_b = get_a((p + "attn_norm.bias").c_str(),   {ah.d_model});
            L.fc1_w = get_a((p + "fc1.weight").c_str(), {ah.d_model, ah.n_ff}); // ne={in,out}
            L.fc1_b = get_a((p + "fc1.bias").c_str(),   {ah.n_ff});
            L.fc2_w = get_a((p + "fc2.weight").c_str(), {ah.n_ff, ah.d_model});
            L.fc2_b = get_a((p + "fc2.bias").c_str(),   {ah.d_model});
            L.final_norm_w = get_a((p + "final_norm.weight").c_str(), {ah.d_model});
            L.final_norm_b = get_a((p + "final_norm.bias").c_str(),   {ah.d_model});

            ok = L.attn_q_w && L.attn_q_b && L.attn_k_w && L.attn_k_b &&
                 L.attn_v_w && L.attn_v_b && L.attn_out_w && L.attn_out_b &&
                 L.attn_norm_w && L.attn_norm_b && L.fc1_w && L.fc1_b &&
                 L.fc2_w && L.fc2_b && L.final_norm_w && L.final_norm_b;
        }
        ok = ok && model.a_conv1_w && model.a_conv1_b && model.a_conv2_w && model.a_conv2_b &&
             model.a_conv3_w && model.a_conv3_b && model.a_conv_out_w &&
             model.a_ln_post_w && model.a_ln_post_b &&
             model.a_proj1_w && model.a_proj1_b && model.a_proj2_w && model.a_proj2_b;

        if (!ok) {
            LOG(ERROR) << __func__ << ": ASR audio tower tensors missing or mis-shaped";
            gguf_free(ctx_gguf);
            return false;
        }
        LOG(INFO) << __func__ << ": ASR audio tower loaded: layers=" << ah.n_audio_layer
                  << " d_model=" << ah.d_model << " heads=" << ah.n_head
                  << " ffn=" << ah.n_ff << " out_dim=" << ah.output_dim
                  << " conv_channels=" << ah.conv_channels
                  << " mel_bins=" << ah.num_mel_bins
                  << " audio_pad_id=" << ah.audio_token_id;
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
