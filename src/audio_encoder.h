#pragma once

// ============================================================
// Qwen3-ASR 音频 Encoder（asr/v0.2）
//
// 输入参考 Mel [num_mel_bins, T]，在 ggml 上复现官方音频塔：
//   3 层 Conv2d(k=3,s=2,p=1) + GELU
//   → channel/frequency 展平 + conv_out 线性投影(7680→896)
//   → 局部正弦位置编码（每个 CNN 块从 0 重新开始）
//   → 去 padding、按时间顺序拼接为 [A,896]
//   → 18 层音频 Transformer（LayerNorm + 非因果 MHA + LayerNorm + FFN，均带 bias）
//   → ln_post + proj1 + GELU + proj2 投影到 [A,1024]
// 输出连续音频特征 [A, output_dim]，供上层替换 audio_pad 位置的文本 Embedding。
//
// 分块与有效长度（对应设计 §3.2 第 3 条）：
//   CNN 每 n_window*2=100 个有效 Mel 帧分一块，尾块 zero-pad 到 100；
//   A(T) = 13*floor(T/100) + ceil((T mod 100)/8)，不能使用整段 ceil(T/8)。
//
// v0.2 约束：单注意力窗口内的短音频（A ≤ 104），此时音频 Transformer 对全部
// A 个位置做全局双向（非因果）注意力；跨窗口块对角 mask 属 v0.4。
//
// 权重张量布局见 src/model.h 的 qwen3_model（asr.audio.* 命名空间）。
// ============================================================

#include "model.h"

#include <cstdint>
#include <vector>

// 音频位置数 A(T) = 13*floor(T/100) + ceil((T mod 100)/8)（官方整数公式）。
int32_t qwen3_asr_audio_positions(int32_t T);

struct qwen3_asr_audio_result {
    int32_t A = 0;                 // 有效音频位置数
    std::vector<float> features;   // [A, output_dim] 行主序（最终投影输出）

    // 可选中间量（want_intermediates=true 时填充），供数值对齐逐段比对：
    std::vector<float> conv_out;   // [A, d_model]               Transformer 输入（CNN + 位置编码，去 padding）
    std::vector<float> hidden;     // [n_audio_layer+1, A, d_model]  hidden[0]=conv_out，hidden[i]=第 i 层输出
};

// 运行音频塔。mel 为 [n_mel_bins, T] 行主序 F32（n_mel_bins 需等于 model.asr_hparams.num_mel_bins）。
// 成功返回 true 并填充 out；失败（形状不符 / 超长 / 显存）返回 false。
bool qwen3_asr_encode_audio(const qwen3_model & model,
                            const float * mel, int32_t T,
                            bool want_intermediates,
                            qwen3_asr_audio_result & out);
