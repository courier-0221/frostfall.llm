#pragma once

// ============================================================
// Qwen3-ASR Log-Mel 音频前处理（asr/v0.3）—— 纯 DSP，无权重
//
// 在音频塔（src/audio_encoder.*，属于模型）之前，把单声道 float32 PCM 变成
// [128,T] 的 Log-Mel 特征，数值契约对齐本模型官方 WhisperFeatureExtractor
// （transformers 4.57.6，torch 路径），实现方式为剥离移植 whisper.cpp 的纯 DSP
// 部分（STFT / filterbank / log 归一化），不整包引入 whisper 推理：
//
//   PCM → 反射填充 n_fft/2（居中 STFT，torch.stft 默认 pad_mode="reflect"）
//       → 周期 Hann 窗（非 Povey；无预加重、无 dither）
//       → 实数 FFT（n_fft=400）→ 功率谱 |X|^2 → Slaney mel filterbank（mel 刻度
//         与面积归一化均为 Slaney，0~sampling_rate/2 Hz）
//       → log10(max(mel_power, 1e-10))
//       → 丢弃最后一帧（T = floor(n_samples/hop)，对应 attention mask 的有效帧数）
//       → max(log, log_max-8) → (log+4)/4
//
// 与官方实现的已知数值差异（记录于 doc/asr/design_asr.md §4.7）：
//   - 官方 torch 路径全程 float32（STFT/filterbank 均 f32 存储）；
//     本实现 filterbank 系数按 f32 量化、窗函数 f32，帧内 FFT 在 double 上
//     完成后量化到 f32 模拟官方 f32 FFT 输出；实测与 golden mel 的
//     max_abs ≈ 2.3e-5、RMSE ≈ 6.2e-7，远小于上层音频塔容差。
//
// 结构参数不硬编码：来自 GGUF `asr.mel.*`（见 src/model.cpp），本结构体的
// 缺省值即本模型契约缺省（16000/400/160/128/right），便于独立单元测试。
// ============================================================

#include <cstdint>
#include <string>
#include <vector>

// Log-Mel 前处理契约（GGUF `asr.mel.*`）。
struct qwen3_asr_mel_hparams {
    int32_t sampling_rate = 16000; // asr.mel.sampling_rate
    int32_t n_fft         = 400;   // asr.mel.n_fft（帧长 = FFT 长度）
    int32_t hop_length    = 160;   // asr.mel.hop_length
    int32_t feature_size  = 128;   // asr.mel.feature_size（mel bin 数）
    std::string padding_side = "right"; // asr.mel.padding_side（当前仅支持 right）
};

struct qwen3_asr_log_mel_result {
    int32_t T = 0;              // 有效 Mel 帧数 = floor(n_samples/hop_length)
    int32_t n_frames_fft = 0;   // STFT 产生的帧数 = T+1（最后 1 帧按官方行为丢弃）
    float log_mel_max = 0.0f;   // 归一化后矩阵最大值（调试输出，= (log10_max+4)/4）
    std::vector<float> mel;     // Log-Mel，布局 [feature_size, T]（mel[bin*T+t]，同 golden npy）
};

// 计算 Log-Mel。pcm 为单声道 float32（建议在 [-1,1]）；n_samples < n_fft、
// 含 NaN/Inf、参数非法时返回 false 并写入 err。
bool qwen3_asr_log_mel(const qwen3_asr_mel_hparams & mp,
                       const float * pcm, int32_t n_samples,
                       qwen3_asr_log_mel_result & out,
                       std::string * err);
