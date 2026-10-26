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
// 窗口语义（asr/v0.4，官方 get_audio_cu_seqlens）：音频 Transformer 对 A 个
// 位置按大小 window_aftercnn = 13*(n_window_infer/100) = 104 的连续窗口做块对角
// 双向注意力（窗口内双向、窗口间隔离；尾块凑满最后窗口）。A <= 104 的单窗口
// 输入与全局双向逐位一致（v0.2/v0.3 行为不变）；A > 窗口容量配额返回失败，由
// 上层（asr_engine）非重叠逐段处理。
//
// 权重张量布局见 src/model.h 的 qwen3_model（asr.audio.* 命名空间）。
// ============================================================

#include "model.h"

#include <cstdint>
#include <functional>
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

// 单请求窗口容量配额（音频位置数）= 13 * (n_window_infer/chunk)。0 = 非 ASR 模型。
int32_t qwen3_asr_max_audio_positions(const qwen3_model & model);

// 运行音频塔。mel 为 [n_mel_bins, T] 行主序 F32（n_mel_bins 需等于 model.asr_hparams.num_mel_bins）。
// 成功返回 true 并填充 out；失败（形状不符 / 超长 / 窗口容量配额 / 显存）返回 false。
// 返回 false 且无错误日志 = on_cancel 置位（调用方按取消语义处理）。
// on_cancel（可选）：CNN 分块批边界查询，true = 立即中止。
// chunk_capacity：一次构图执行的 CNN 块数上限（<=0 = 不分批）；分批不改变数值。
bool qwen3_asr_encode_audio(const qwen3_model & model,
                            const float * mel, int32_t T,
                            bool want_intermediates,
                            qwen3_asr_audio_result & out,
                            const std::function<bool()> & on_cancel = nullptr,
                            int32_t chunk_capacity = -1);
