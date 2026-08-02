#pragma once

// frostfall.embed v0.1 —— pooling / MRL 截断 / L2 归一化。
//
// 输入：模型最后一层的隐层张量 hidden[n_embd, n_tokens]（行主序，同 ggml 内存布局）。
// 输出：单条 embedding 向量 std::vector<float>，长度 = target_dim（或 n_embd）。
//
// 处理顺序（顺序不能反）：
//   1. last-token pool：取最后一列 hidden[:, n_tokens-1]，得 [n_embd] 向量。
//      Qwen3-Embedding 官方用 last-token pool（1_Pooling/config.json：pooling_mode_lasttoken=true）。
//      v0.1 无 padding，"最后一列"就是序列末端；v1.0 batch 场景改为左 padding 后仍等价。
//   2. MRL 截断：target_dim ∈ [32, n_embd]，截断前 target_dim 维；target_dim <= 0 或
//      >= n_embd 时不截断。
//   3. L2 归一化：v[i] /= ‖v‖₂，保证输出是单位向量。
//      ★ 必须在截断之后再归一化，先归一化再截断会失去单位向量性质。

#include <cstdint>
#include <vector>

// 从 hidden[n_embd, n_tokens] 中提取单条 embedding：last-token pool -> MRL 截断 -> L2 归一化。
//
// hidden_data：ggml 张量的内存指针（type=F32，layout=[n_embd, n_tokens] 行主序）。
// target_dim： <= 0 或 >= n_embd 时取满维 n_embd；否则截断到 target_dim。
std::vector<float> qwen3_embed_pool_and_normalize(
        const float * hidden_data,
        int32_t       n_embd,
        int32_t       n_tokens,
        int32_t       target_dim);
