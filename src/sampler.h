#pragma once

// frostfall v0.3 —— 采样策略（sampler）。
//
// v0.2 只做贪心解码（argmax）；v0.3 在此基础上支持一条常见的采样链：
//   repeat penalty -> top-k -> temperature+softmax -> top-p -> 按概率随机采样。
// 各步都可单独关闭：
//   - temp <= 0            退化为贪心（argmax），此时其它参数被忽略，输出与 v0.2 完全一致。
//   - top_k <= 0           不做 top-k 截断（保留全部候选）。
//   - top_p >= 1.0         不做 top-p（nucleus）截断。
//   - repeat_penalty == 1  不做重复惩罚。
//
// 固定 seed 时输出可复现（用 std::mt19937 作为随机源）。

#include <cstdint>
#include <random>
#include <vector>

// 特殊 seed：表示“未指定，运行时随机取一个”。resolve 后会回填成真正用到的值以便打印复现。
constexpr uint32_t SAMPLER_SEED_RANDOM = 0xFFFFFFFFu;

struct sampler_params {
    float    temp           = 0.0f;                // <=0 => 贪心；否则 logits /= temp
    int32_t  top_k          = 0;                   // <=0 => 关闭（保留全部候选）
    float    top_p          = 1.0f;                // >=1 => 关闭（nucleus 采样阈值）
    float    repeat_penalty = 1.0f;                // ==1 => 关闭；>1 抑制近期出现过的 token
    int32_t  repeat_last_n  = 64;                  // 重复惩罚回看窗口；<0 表示整段历史
    uint32_t seed           = SAMPLER_SEED_RANDOM; // SAMPLER_SEED_RANDOM => 运行时随机
};

struct sampler {
    sampler_params params;
    std::mt19937   rng;

    // 用参数初始化；若 seed 为 SAMPLER_SEED_RANDOM 则从 std::random_device 取一个并回填 params.seed。
    void init(const sampler_params & p);

    // 从最后一个位置的 logits 里采样出下一个 token id。
    //   logits  : 长度为 n_vocab，可被本函数原地修改（会施加重复惩罚）。
    //   prev    : 到目前为止的完整 token 序列（用于重复惩罚的回看窗口）。
    int32_t sample(float * logits, int32_t n_vocab, const std::vector<int32_t> & prev);
};
