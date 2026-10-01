#include "audio_encoder.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "log.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// 官方整数公式：A(T) = 13*floor(T/100) + ceil((T mod 100)/8)。
// 单个长度 len 的 CNN 块产出的有效位置数 = ceil(len/8) = (len+7)/8（len=100 -> 13）。
int32_t qwen3_asr_audio_positions(int32_t T) {
    if (T <= 0) return 0;
    return 13 * (T / 100) + ((T % 100) + 7) / 8;
}

namespace {

// 音频塔结构参数（从 model.asr_hparams 取，避免硬编码）。
struct AudioCfg {
    int32_t d_model;      // 896
    int32_t n_head;       // 14
    int32_t n_layer;      // 18
    int32_t n_ff;         // 3584
    int32_t output_dim;   // 1024
    int32_t num_mel_bins; // 128
    int32_t chunk;        // CNN 分块帧数 = n_window*2 = 100
    int32_t conv_c;       // CNN 通道数 480
    int32_t max_pos;      // 正弦位置表长度 1500
    float   eps;          // LayerNorm eps 1e-5
    int32_t conv_time_out; // 单块卷积后时间维 = 13
};

AudioCfg make_cfg(const qwen3_asr_hparams & ah) {
    AudioCfg c;
    c.d_model      = ah.d_model;
    c.n_head       = ah.n_head;
    c.n_layer      = ah.n_audio_layer;
    c.n_ff         = ah.n_ff;
    c.output_dim   = ah.output_dim;
    c.num_mel_bins = ah.num_mel_bins;
    c.chunk        = ah.n_window * 2;
    c.conv_c       = ah.conv_channels;
    c.max_pos      = ah.max_source_positions;
    c.eps          = ah.layer_norm_eps;
    // 频率轴 128 -> 64 -> 32 -> 16；时间轴 chunk -> ... 每层 (x+2*1-3)/2+1
    auto conv_len = [&](int32_t in) {
        return (in + 2 * ah.conv_padding - ah.conv_kernel) / ah.conv_stride + 1;
    };
    c.conv_time_out = conv_len(conv_len(conv_len(c.chunk))); // 100->50->25->13
    return c;
}

// 局部正弦位置编码（官方 SinusoidsPositionEmbedding，公式生成，非持久 buffer）。
// 返回 [length, d_model] 行主序（每个位置一行 896 个值，即 ggml ne0=d_model 的
// 连续内存布局），可直接 backend_tensor_set 到 [d_model, length] 张量并广播到各 CNN 块。
void sinusoids_f32(int32_t length, int32_t d_model, float max_timescale, std::vector<float> & out) {
    out.assign((size_t) d_model * length, 0.0f);
    const int32_t half = d_model / 2;
    const float inc = std::log(max_timescale) / (float) (half - 1);
    for (int32_t p = 0; p < length; ++p) {
        for (int32_t j = 0; j < half; ++j) {
            const float inv = std::exp(-inc * (float) j);
            const float scaled = (float) p * inv;
            // 行主序 [length, d_model]：位置 p 的第 j / (j+half) 个通道
            out[(size_t) p * d_model + j]          = std::sin(scaled);  // 前半 sin
            out[(size_t) p * d_model + (j + half)]  = std::cos(scaled); // 后半 cos
        }
    }
}

// F16 权重 -> F32 临时张量（im2col 的 dst_type 跟随 kernel 类型，转 F32 避免激活被量化）。
struct ggml_tensor * cast_f32(struct ggml_context * ctx, struct ggml_tensor * w) {
    struct ggml_tensor * d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
    return ggml_cpy(ctx, w, d);
}

constexpr int kMaxNodes = 4096;

struct ggml_init_params make_init_params() {
    struct ggml_init_params p = {
        /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return p;
}

// ---- 阶段 1：分块 Conv2d×3 + GELU + 展平 + conv_out + 局部正弦位置编码 ----
// 输入 mel [num_mel_bins, T]（host），输出 [d_model, conv_time_out, N]（host，未去 padding）。
bool run_cnn_stage(const qwen3_model & model, const AudioCfg & cfg,
                   const float * mel, int32_t T, int32_t N,
                   ggml_backend_t backend, ggml_gallocr_t allocr,
                   std::vector<float> & stage1 /* [d_model, conv_time_out, N] */) {
    struct ggml_context * ctx = ggml_init(make_init_params());
    if (!ctx) return false;

    const int32_t mel_bins = cfg.num_mel_bins;
    const int32_t chunk    = cfg.chunk;

    // 数据张量 b：[W=time(chunk), H=freq(mel_bins), C=1, N]
    struct ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, chunk, mel_bins, 1, N);
    ggml_set_name(x, "mel_in");
    ggml_set_input(x);

    // conv1/2/3（kernel cast 到 F32，im2col 输出 F32，全链路激活保持 F32）
    struct ggml_tensor * w1 = cast_f32(ctx, model.a_conv1_w);
    struct ggml_tensor * w2 = cast_f32(ctx, model.a_conv2_w);
    struct ggml_tensor * w3 = cast_f32(ctx, model.a_conv3_w);
    const int s = model.asr_hparams.conv_stride;
    const int p = model.asr_hparams.conv_padding;

    // conv 输出布局为 [OW=time, OH=freq, OC=channel, N]，bias 是逐通道的（作用在 ne2）。
    // ggml_add 按 ggml_can_repeat 广播，需把 bias reshape 成 [1,1,OC,1] 才能在 time/freq/batch 上广播。
    auto conv_bias = [&](struct ggml_tensor * b) -> struct ggml_tensor * {
        return ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    };

    // 官方为 gelu(conv2d(x, w, b))：bias 属于卷积的一部分，必须在 GELU 之前加。
    // ggml_conv_2d 不带 bias，故先 add 再 gelu，顺序错了会污染整条音频塔。
    // ggml_conv_2d(a=kernel[KW,KH,IC,OC], b=data[W,H,C,N], s0(time),s1(freq), p0,p1, d0,d1)
    // 输出 [OW=time, OH=freq, OC, N]
    struct ggml_tensor * c1 = ggml_conv_2d(ctx, w1, x,  s, s, p, p, 1, 1); // [50,64,480,N]
    c1 = ggml_add(ctx, c1, conv_bias(model.a_conv1_b)); // bias [1,1,480,1] 广播到 ne2
    c1 = ggml_gelu(ctx, c1);

    struct ggml_tensor * c2 = ggml_conv_2d(ctx, w2, c1, s, s, p, p, 1, 1); // [25,32,480,N]
    c2 = ggml_add(ctx, c2, conv_bias(model.a_conv2_b));
    c2 = ggml_gelu(ctx, c2);

    struct ggml_tensor * c3 = ggml_conv_2d(ctx, w3, c2, s, s, p, p, 1, 1); // [13,16,480,N]
    c3 = ggml_add(ctx, c3, conv_bias(model.a_conv3_b));
    c3 = ggml_gelu(ctx, c3);

    // channel/frequency 展平：c3 [time=13, freq=16, ch=480, N]
    // 目标：把 (freq 最快、ch 次之) 展平为 7680（= 官方 permute(0,3,1,2).view(b,t,c*f)，索引 c*16+f）。
    // ggml_permute(a,ax0..3) 把 a 的第 i 轴放到结果 ax_i 位置，故 permute(2,0,1,3)：
    //   c3 轴0(time)->位2，轴1(freq)->位0，轴2(ch)->位1 => [freq=16, ch=480, time=13, N]
    // cont 后内存顺序 freq 最快、ch 次之，reshape 合并为 [freq*ch, time, N]。
    struct ggml_tensor * flat = ggml_permute(ctx, c3, 2, 0, 1, 3);
    flat = ggml_cont(ctx, flat);
    flat = ggml_reshape_3d(ctx, flat, model.a_conv_out_w->ne[0], cfg.conv_time_out, N); // [7680, 13, N]

    // conv_out 线性投影（无 bias）：a_conv_out_w [7680, 896]
    struct ggml_tensor * co = ggml_mul_mat(ctx, model.a_conv_out_w, flat); // [896, 13, N]

    // 局部正弦位置编码：每块用相同的 pos[:13]，广播到 N
    std::vector<float> pos;
    sinusoids_f32(cfg.conv_time_out, cfg.d_model, 10000.0f, pos);
    struct ggml_tensor * pos_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.d_model, cfg.conv_time_out, 1);
    ggml_set_name(pos_t, "pos");
    ggml_set_input(pos_t);
    co = ggml_add(ctx, co, pos_t); // [896,13,N] + [896,13,1] 广播

    ggml_set_output(co);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_build_forward_expand(gf, co);

    ggml_gallocr_alloc_graph(allocr, gf);

    // 填充 mel（尾块 zero-pad 到 chunk）
    std::vector<float> mel_buf((size_t) chunk * mel_bins * N, 0.0f);
    for (int32_t n = 0; n < N; ++n) {
        for (int32_t f = 0; f < mel_bins; ++f) {
            for (int32_t t = 0; t < chunk; ++t) {
                const int32_t tg = n * chunk + t;
                if (tg < T) {
                    mel_buf[((size_t) n * mel_bins + f) * chunk + t] = mel[(size_t) f * T + tg];
                }
            }
        }
    }
    ggml_backend_tensor_set(x, mel_buf.data(), 0, mel_buf.size() * sizeof(float));
    ggml_backend_tensor_set(pos_t, pos.data(), 0, pos.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    stage1.assign((size_t) cfg.d_model * cfg.conv_time_out * N, 0.0f);
    ggml_backend_tensor_get(co, stage1.data(), 0, stage1.size() * sizeof(float));

    ggml_free(ctx);
    return true;
}

// ---- 阶段 2：18 层音频 Transformer + 输出投影 ----
// 输入 conv_out [A, d_model]（host，已去 padding、按块拼接），输出 features [A, output_dim]。
bool run_transformer_stage(const qwen3_model & model, const AudioCfg & cfg,
                           const std::vector<float> & conv_out, int32_t A,
                           bool want_intermediates,
                           ggml_backend_t backend, ggml_gallocr_t allocr,
                           std::vector<float> & features,   // [A, output_dim]
                           std::vector<float> & hidden_all) {// [n_layer+1, A, d_model]
    struct ggml_context * ctx = ggml_init(make_init_params());
    if (!ctx) return false;

    const int64_t d_model = cfg.d_model;
    const int64_t n_head  = cfg.n_head;
    const int64_t head_dim = d_model / n_head; // 64
    const float scale = 1.0f / std::sqrt((float) head_dim);

    // 输入 [d_model, A]
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, A);
    ggml_set_name(x, "audio_in");
    ggml_set_input(x);

    std::vector<struct ggml_tensor *> layer_outs;

    struct ggml_tensor * cur = x;
    for (int32_t il = 0; il < cfg.n_layer; ++il) {
        const qwen3_asr_audio_layer & L = model.asr_audio_layers[il];

        // ---- 自注意力（非因果、无 mask、块内双向）----
        struct ggml_tensor * residual = cur;
        struct ggml_tensor * h = ggml_norm(ctx, cur, cfg.eps);   // LayerNorm（无仿射）
        h = ggml_mul(ctx, h, L.attn_norm_w);
        h = ggml_add(ctx, h, L.attn_norm_b);

        struct ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, L.attn_q_w, h), L.attn_q_b); // [896,A]
        struct ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, L.attn_k_w, h), L.attn_k_b);
        struct ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, L.attn_v_w, h), L.attn_v_b);

        // reshape 到 [head_dim, n_head, A]
        q = ggml_reshape_3d(ctx, q, head_dim, n_head, A);
        k = ggml_reshape_3d(ctx, k, head_dim, n_head, A);
        v = ggml_reshape_3d(ctx, v, head_dim, n_head, A);

        // K^T Q：permute 成 [head_dim, A, n_head] 后 mul_mat -> [A_k, A_q, n_head]
        struct ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3); // [head_dim, A, n_head]
        struct ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3); // [head_dim, A, n_head]
        struct ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);        // [A_k, A_q, n_head]
        kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);

        // V 转成 [A_k, head_dim, n_head]，kqv = V^T softmax(kq) -> [head_dim, A_q, n_head]
        // 注：ggml_permute(a, ax0,ax1,ax2,ax3) 把 a 的第 i 轴放到结果的 ax_i 位置；
        //     v=[head_dim,n_head,A] -> [A,head_dim,n_head] 需 permute(1,2,0,3)。
        struct ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [A, head_dim, n_head]
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, vp, kq);                       // [head_dim, A_q, n_head]

        // 合并头 -> [d_model, A]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);              // [head_dim, n_head, A]
        struct ggml_tensor * att = ggml_cont_2d(ctx, kqv, d_model, A);
        att = ggml_add(ctx, ggml_mul_mat(ctx, L.attn_out_w, att), L.attn_out_b);
        cur = ggml_add(ctx, residual, att);

        // ---- FFN（LayerNorm -> fc1 -> GELU -> fc2，均带 bias）----
        residual = cur;
        h = ggml_norm(ctx, cur, cfg.eps);
        h = ggml_mul(ctx, h, L.final_norm_w);
        h = ggml_add(ctx, h, L.final_norm_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, L.fc1_w, h), L.fc1_b); // [n_ff, A]
        h = ggml_gelu(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, L.fc2_w, h), L.fc2_b); // [d_model, A]
        cur = ggml_add(ctx, residual, h);

        if (want_intermediates) {
            struct ggml_tensor * lo = ggml_cont(ctx, cur);
            ggml_set_output(lo);
            layer_outs.push_back(lo);
        }
    }

    // ---- 输出投影：ln_post -> proj1 -> GELU -> proj2 ----
    struct ggml_tensor * o = ggml_norm(ctx, cur, cfg.eps);
    o = ggml_mul(ctx, o, model.a_ln_post_w);
    o = ggml_add(ctx, o, model.a_ln_post_b);
    o = ggml_add(ctx, ggml_mul_mat(ctx, model.a_proj1_w, o), model.a_proj1_b); // [896, A]
    o = ggml_gelu(ctx, o);
    o = ggml_add(ctx, ggml_mul_mat(ctx, model.a_proj2_w, o), model.a_proj2_b); // [1024, A]
    ggml_set_name(o, "audio_features");
    ggml_set_output(o);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_build_forward_expand(gf, o);
    // 逐层输出节点不在 o 的上游链路上，必须单独 expand 才会进入图并被 gallocr 分配 buffer。
    for (struct ggml_tensor * lo : layer_outs) ggml_build_forward_expand(gf, lo);

    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(x, conv_out.data(), 0, (size_t) d_model * A * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    features.assign((size_t) A * cfg.output_dim, 0.0f);
    ggml_backend_tensor_get(o, features.data(), 0, features.size() * sizeof(float));

    if (want_intermediates) {
        hidden_all.assign((size_t) (cfg.n_layer + 1) * A * d_model, 0.0f);
        // hidden[0] = conv_out（Transformer 输入）
        std::memcpy(hidden_all.data(), conv_out.data(), (size_t) A * d_model * sizeof(float));
        for (int32_t il = 0; il < cfg.n_layer; ++il) {
            float * dst = hidden_all.data() + (size_t) (il + 1) * A * d_model;
            ggml_backend_tensor_get(layer_outs[il], dst, 0, (size_t) A * d_model * sizeof(float));
        }
    }

    ggml_free(ctx);
    return true;
}

} // namespace

bool qwen3_asr_encode_audio(const qwen3_model & model,
                            const float * mel, int32_t T,
                            bool want_intermediates,
                            qwen3_asr_audio_result & out) {
    if (!model.is_asr) {
        LOG(ERROR) << __func__ << ": model is not qwen3-asr";
        return false;
    }
    if (!mel || T <= 0) {
        LOG(ERROR) << __func__ << ": invalid mel input (T=" << T << ")";
        return false;
    }
    const AudioCfg cfg = make_cfg(model.asr_hparams);
    if (cfg.num_mel_bins <= 0 || cfg.d_model <= 0 || cfg.n_head <= 0) {
        LOG(ERROR) << __func__ << ": incomplete audio hparams";
        return false;
    }

    const int32_t N = (T + cfg.chunk - 1) / cfg.chunk; // CNN 块数 = ceil(T/100)
    const int32_t A = qwen3_asr_audio_positions(T);
    // v0.2 单注意力窗口约束：A <= 13*(n_window_infer/chunk)
    const int32_t a_max = 13 * (model.asr_hparams.n_window_infer / cfg.chunk);
    if (A > a_max) {
        LOG(ERROR) << __func__ << ": audio too long for v0.2 single-window encoder (A=" << A
                   << " > " << a_max << "); cross-window support is asr/v0.4";
        return false;
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) return false;

    // ---- 阶段 1：CNN ----
    std::vector<float> stage1; // [d_model, conv_time_out, N]
    if (!run_cnn_stage(model, cfg, mel, T, N, model.backend, allocr, stage1)) {
        ggml_gallocr_free(allocr);
        return false;
    }

    // ---- 去 padding、按块顺序拼接为 [A, d_model] ----
    // stage1 内存布局：element(d, p, n) 位于 d + p*d_model + n*d_model*conv_time_out
    std::vector<float> conv_out((size_t) A * cfg.d_model, 0.0f);
    {
        int32_t row = 0;
        for (int32_t n = 0; n < N; ++n) {
            const int32_t len = std::min(cfg.chunk, T - n * cfg.chunk);
            const int32_t valid = (len + 7) / 8; // = ceil(len/8)；len=100 -> 13
            for (int32_t p = 0; p < valid; ++p, ++row) {
                const float * src = stage1.data() + (size_t) n * cfg.d_model * cfg.conv_time_out
                                    + (size_t) p * cfg.d_model;
                std::memcpy(conv_out.data() + (size_t) row * cfg.d_model, src,
                            (size_t) cfg.d_model * sizeof(float));
            }
        }
        if (row != A) {
            LOG(ERROR) << __func__ << ": gathered " << row << " positions, expected A=" << A;
            ggml_gallocr_free(allocr);
            return false;
        }
    }

    // ---- 阶段 2：Transformer + 投影 ----
    std::vector<float> features, hidden_all;
    if (!run_transformer_stage(model, cfg, conv_out, A, want_intermediates,
                               model.backend, allocr, features, hidden_all)) {
        ggml_gallocr_free(allocr);
        return false;
    }

    ggml_gallocr_free(allocr);

    out.A = A;
    out.features = std::move(features);
    if (want_intermediates) {
        out.conv_out = std::move(conv_out);
        out.hidden   = std::move(hidden_all);
    }
    return true;
}
