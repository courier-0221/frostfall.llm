#include "log_mel.h"

// ============================================================
// Log-Mel 纯 DSP 实现（asr/v0.3）
//
// 参考与差异（详见 src/log_mel.h 头注释与 doc/asr/design_asr.md §4.3/§4.7）：
//   - STFT：居中 + 反射填充 + 周期 Hann，行为对齐 torch.stft（transformers
//     WhisperFeatureExtractor._torch_extract_fbank_features）；
//   - FFT：剥离移植 whisper.cpp 的 Cooley-Tukey（whisper.cpp/src/whisper.cpp
//     fft()/dft()），double 精度 + 输出量化 f32，替代官方 pocketfft-f32；
//   - filterbank：运行时生成 librosa/Slaney 公式（mel 刻度 + 面积归一化均
//     Slaney，fmin=0、fmax=sampling_rate/2），系数量化 f32——
//     与 transformers.audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney")
//     逐系数一致（已用 numpy 验证 diff=0）；
//   - 归一化：log10(max(·,1e-10)) → 丢弃最后 1 帧 → max(·, max-8) → (·+4)/4，
//     全程 f32，与官方一致。
// ============================================================

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

// ---------- Slaney mel 刻度（transformers.audio_utils.hertz_to_mel / mel_to_hertz，mel_scale="slaney"） ----------

double hz_to_mel_slaney(double f) {
    if (f >= 1000.0) {
        constexpr double kLogstep = 27.0 / std::log(6.4);
        return 15.0 + std::log(f / 1000.0) * kLogstep;
    }
    return 3.0 * f / 200.0;
}

double mel_to_hz_slaney(double m) {
    if (m >= 15.0) {
        constexpr double kLogstep = std::log(6.4) / 27.0;
        return 1000.0 * std::exp(kLogstep * (m - 15.0));
    }
    return 200.0 * m / 3.0;
}

// 生成 Slaney filterbank：[n_freq_bins, n_mels]，与 mel_filter_bank(norm="slaney",
// mel_scale="slaney") 逐系数一致；fft 频点为 linspace(0, sr/2, n_freq_bins)。
// 注意官方用 sr//2（整除）作 fmax，此处保持一致。
std::vector<double> slaney_filterbank(int32_t n_freq_bins, int32_t n_mels, int32_t sr) {
    const double fmin = 0.0;
    const double fmax = (double) (sr / 2);
    std::vector<double> mel_f((size_t) n_mels + 2);
    for (int32_t i = 0; i <= n_mels + 1; ++i) {
        const double m_lo = hz_to_mel_slaney(fmin);
        const double m_hi = hz_to_mel_slaney(fmax);
        const double m = m_lo + (m_hi - m_lo) * (double) i / (double) (n_mels + 1);
        mel_f[(size_t) i] = mel_to_hz_slaney(m);
    }
    // fft 频点：bin k 频率 = linspace(0, sr/2, n_freq_bins) => k * (sr/2) / (n_freq_bins-1)
    const double bin_hz = fmax / (double) (n_freq_bins - 1);

    std::vector<double> fb((size_t) n_freq_bins * n_mels, 0.0);
    for (int32_t j = 0; j < n_mels; ++j) {
        // 三角窗（librosa/HF 一致）：对频点 fk，
        //   down = (fk - mel_f[j]) / (mel_f[j+1] - mel_f[j])            （左上升沿）
        //   up   = (mel_f[j+2] - fk) / (mel_f[j+2] - mel_f[j+1])        （右下降沿）
        //   w    = max(0, min(down, up)) * enorm
        const double fdiff_lo = mel_f[(size_t) j + 1] - mel_f[(size_t) j + 0];
        const double fdiff_hi = mel_f[(size_t) j + 2] - mel_f[(size_t) j + 1];
        // Slaney 面积归一化：enorm = 2 / (mel_f[j+2] - mel_f[j])
        const double enorm = 2.0 / (mel_f[(size_t) j + 2] - mel_f[(size_t) j + 0]);
        for (int32_t k = 0; k < n_freq_bins; ++k) {
            const double fk = bin_hz * (double) k;
            const double down = -(mel_f[(size_t) j + 0] - fk) / fdiff_lo;
            const double up    = (mel_f[(size_t) j + 2] - fk) / fdiff_hi;
            const double w = std::max(0.0, std::min(down, up)) * enorm;
            fb[(size_t) j * n_freq_bins + (size_t) k] = w;
        }
    }
    return fb; // [n_mels, n_freq_bins] 行主序
}

// ---------- FFT：剥离移植 whisper.cpp fft()/dft()，float→double，精度 double ----------

struct SinCosTable {
    std::vector<double> sin_v, cos_v;
    explicit SinCosTable(int n) : sin_v((size_t) n), cos_v((size_t) n) {
        for (int i = 0; i < n; ++i) {
            const double theta = (2.0 * M_PI * i) / n;
            sin_v[(size_t) i] = std::sin(theta);
            cos_v[(size_t) i] = std::cos(theta);
        }
    }
};

// 每个 n_fft 一张表（线程语义：引擎 Init/Transcribe 串行互斥，见 asr_engine.h）。
const SinCosTable & sincos_table(int n_fft) {
    static std::vector<std::pair<int, SinCosTable *>> cache;
    for (auto & kv : cache) if (kv.first == n_fft) return *kv.second;
    cache.emplace_back(n_fft, new SinCosTable(n_fft));
    return *cache.back().second;
}

// 朴素 DFT（输入为 interleave{re,im}，输出同样 interleave）；仅用于小奇数 N。
void dft(const double * in, int N, int n_fft, double * out) {
    const SinCosTable & t = sincos_table(n_fft);
    const int step = n_fft / N;
    for (int k = 0; k < N; ++k) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; ++n) {
            const int idx = (k * n * step) % n_fft; // theta = 2*PI*k*n/N
            re += in[2 * n + 0] * t.cos_v[(size_t) idx];
            im -= in[2 * n + 0] * t.sin_v[(size_t) idx];
        }
        out[2 * k + 0] = re;
        out[2 * k + 1] = im;
    }
}

// Cooley-Tukey 递归（同 whisper.cpp 布局约定：偶/奇序列复用 in+N 做滚动 scratch）。
// in 需要至少 4N 个 double 的缓冲（whisper.cpp 为 2N 输入 + fft 内部复用，
// 这里显式由调用方提供 scratch）。
void fft_rec(double * in, int N, int n_fft, double * scratch) {
    if (N == 1) {
        scratch[0] = in[0];
        scratch[1] = 0.0;
        return;
    }
    const int half_N = N / 2;
    if (N - half_N * 2 == 1) { // 奇数因子
        dft(in, N, n_fft, scratch);
        return;
    }

    double * even = in + N * 2; // 借用输入缓冲后面的空间放置偶序列（长度 2*half_N）
    for (int i = 0; i < half_N; ++i) {
        even[2 * i + 0] = in[2 * (2 * i) + 0];
        even[2 * i + 1] = in[2 * (2 * i) + 1];
    }
    double * even_fft = scratch + 2 * N;
    fft_rec(even, half_N, n_fft, even_fft);

    // 奇序列放在偶序列同一个缓冲（互斥使用）
    double * odd = in + N * 2;
    for (int i = 0; i < half_N; ++i) {
        odd[2 * i + 0] = in[2 * (2 * i + 1) + 0];
        odd[2 * i + 1] = in[2 * (2 * i + 1) + 1];
    }
    double * odd_fft = even_fft + N * 2;
    fft_rec(odd, half_N, n_fft, odd_fft);

    const SinCosTable & t = sincos_table(n_fft);
    const int step = n_fft / N;
    for (int k = 0; k < half_N; ++k) {
        const int idx = k * step; // theta = 2*PI*k/N
        const double re =  t.cos_v[(size_t) idx];
        const double im = -t.sin_v[(size_t) idx];

        const double re_odd = odd_fft[2 * k + 0];
        const double im_odd = odd_fft[2 * k + 1];

        scratch[2 * k + 0]                  = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        scratch[2 * k + 1]                  = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        scratch[2 * (k + half_N) + 0]       = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        scratch[2 * (k + half_N) + 1]       = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

} // namespace

bool qwen3_asr_log_mel(const qwen3_asr_mel_hparams & mp,
                       const float * pcm, int32_t n_samples,
                       qwen3_asr_log_mel_result & out,
                       std::string * err) {
    auto fail = [&](const std::string & msg) {
        if (err) *err = msg;
        return false;
    };

    out = qwen3_asr_log_mel_result();

    // ---- 参数与输入校验（错误一律显式返回，不静默截断） ----
    if (!pcm && n_samples > 0) return fail("pcm is null");
    if (n_samples <= 0)        return fail("pcm is empty (n_samples=" + std::to_string(n_samples) + ")");
    if (mp.n_fft < 2)                    return fail("mel.n_fft must be >= 2");
    if (mp.hop_length <= 0)              return fail("mel.hop_length must be > 0");
    if (mp.feature_size < 1)             return fail("mel.feature_size must be >= 1");
    if (mp.sampling_rate < 2 * mp.n_fft) return fail("mel.sampling_rate too low for n_fft");
    if (mp.padding_side != "right")
        return fail("mel.padding_side '" + mp.padding_side + "' not supported (only 'right')");
    if (n_samples < mp.n_fft)
        return fail("pcm too short: n_samples=" + std::to_string(n_samples) +
                    " < n_fft=" + std::to_string(mp.n_fft));
    for (int32_t i = 0; i < n_samples; ++i) {
        if (!std::isfinite(pcm[i]))
            return fail("pcm contains NaN/Inf at sample " + std::to_string(i));
    }

    const int32_t n_fft    = mp.n_fft;
    const int32_t hop      = mp.hop_length;
    const int32_t n_mels   = mp.feature_size;
    const int32_t n_bins   = n_fft / 2 + 1; // 单边功率谱 bin 数（k=0..n_fft/2）
    const int32_t pad      = n_fft / 2;     // 居中 STFT 反射填充宽度

    // 反射填充长度必须有效（reflect 镜像要求 pad < n_samples）
    if (pad >= n_samples)
        return fail("pcm too short for reflect padding: n_samples=" + std::to_string(n_samples) +
                    " <= pad=" + std::to_string(pad));

    // ---- 周期 Hann 窗（f32，周期形式，0.5*(1-cos(2*pi*i/n_fft))） ----
    std::vector<float> hann((size_t) n_fft);
    for (int32_t i = 0; i < n_fft; ++i)
        hann[(size_t) i] = (float) (0.5 * (1.0 - std::cos(2.0 * M_PI * i / n_fft)));

    // ---- Slaney filterbank（double 生成 -> f32 量化，对齐官方 f32 存储） ----
    const std::vector<double> fb64 = slaney_filterbank(n_bins, n_mels, mp.sampling_rate); // [n_mels, n_bins]
    std::vector<float> fb32(fb64.size());
    for (size_t i = 0; i < fb64.size(); ++i) fb32[i] = (float) fb64[i];

    // ---- 帧数（含最后 1 帧；官方随后丢弃最后 1 帧） ----
    const int32_t n_frames_fft = 1 + (n_samples / hop); // = 1 + floor(n/hop)
    const int32_t T            = n_frames_fft - 1;      // 有效帧 = floor(n/hop)

    // ---- 逐帧 STFT：反射填充 + 加窗 + FFT + 功率谱 -> mel ----
    // FFT 递归缓冲：fft_rec 的布局约定需要 in 侧 4*n_fft、scratch 侧 8*n_fft 的余量。
    std::vector<double> fft_in((size_t) n_fft * 8, 0.0);
    std::vector<double> fft_scratch((size_t) n_fft * 16, 0.0);
    std::vector<float>  power_bin((size_t) n_bins); // 按 bin 逐帧暂存功率谱

    std::vector<float> log_mel((size_t) n_mels * (size_t) T, 0.0f); // [n_mels, T]

    for (int32_t t = 0; t < T + 1; ++t) {
        const int32_t base = t * hop; // 加窗帧在填充后序列中的起点
        for (int32_t j = 0; j < n_fft; ++j) {
            // 居中 STFT：帧 t 覆盖填充序列 [t*hop, t*hop+n_fft)，对应原始序列索引 base+j-pad
            const int32_t idx = base + j - pad;
            double x;
            if (idx < 0)               x = (double) pcm[-idx];                              // 左镜像（不重复边界样本）
            else if (idx >= n_samples) x = (double) pcm[2 * (n_samples - 1) - idx];        // 右镜像
            else                       x = (double) pcm[idx];
            fft_in[2 * (size_t) j + 0] = (double) (float) (x * (double) hann[(size_t) j]); // f32×f32 语义（量化同官方 f32 加窗帧）
            fft_in[2 * (size_t) j + 1] = 0.0;
        }

        fft_rec(fft_in.data(), n_fft, n_fft, fft_scratch.data());

        // 功率谱 |X|^2（double 计算，量化 f32 对齐官方 abs()**2 的 f32 存储）
        for (int32_t k = 0; k < n_bins; ++k) {
            const double x_k = fft_scratch[2 * (size_t) k + 0];
            const double y_k = fft_scratch[2 * (size_t) k + 1];
            power_bin[(size_t) k] = (float) (x_k * x_k + y_k * y_k);
        }

        if (t < T) { // 最后一帧（t==T）计算后丢弃，与官方 stft[..., :-1] 一致
            for (int32_t j = 0; j < n_mels; ++j) {
                double sum = 0.0; // f32 系数 × f32 功率，double 累加（对齐 f32 matmul 的舍入级别）
                const float * row = fb32.data() + (size_t) j * n_bins;
                for (int32_t k = 0; k < n_bins; ++k)
                    sum += (double) row[k] * (double) power_bin[(size_t) k];
                const float mel_v = std::max((float) 1e-10, (float) sum);
                log_mel[(size_t) j * (size_t) T + (size_t) t] = (float) std::log10((double) mel_v);
            }
        }
    }

    // ---- 丢弃最后 1 帧后的 max-8 截断 + (x+4)/4 归一化（全程 f32） ----
    float max_v = -INFINITY;
    for (float v : log_mel) max_v = std::max(max_v, v);
    const float floor_v = max_v - 8.0f;
    for (float & v : log_mel) {
        v = std::max(v, floor_v);
        v = (v + 4.0f) / 4.0f;
    }
    float max_final = -INFINITY;
    for (float v : log_mel) max_final = std::max(max_final, v);

    out.T            = T;
    out.n_frames_fft = n_frames_fft;
    out.log_mel_max  = max_final;
    out.mel          = std::move(log_mel);
    return true;
}
