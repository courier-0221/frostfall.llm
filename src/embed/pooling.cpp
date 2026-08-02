#include "pooling.h"

#include <algorithm>
#include <cmath>

// ggml 张量 [n_embd, n_tokens] 的内存布局：ne[0]=n_embd 是最快变化维度，
// 每个 token 在内存里占连续 n_embd 个 float，token t 的第 d 维 = hidden_data[t*n_embd + d]。
//
// last-token pool 取最后一个 token（t = n_tokens - 1）的整段向量。
std::vector<float> qwen3_embed_pool_and_normalize(
        const float * hidden_data,
        int32_t       n_embd,
        int32_t       n_tokens,
        int32_t       target_dim) {
    // 1) last-token pool -> [n_embd]
    const float * last = hidden_data + (int64_t)(n_tokens - 1) * n_embd;
    std::vector<float> v(last, last + n_embd);

    // 2) MRL 截断：target_dim ∈ [32, n_embd]；<=0 或 >= n_embd 视为不截断。
    if (target_dim > 0 && target_dim < n_embd) {
        v.resize(target_dim);
    }

    // 3) L2 归一化：v /= sqrt(sum(v^2))
    double sq = 0.0;
    for (float x : v) {
        sq += (double)x * (double)x;
    }
    const float norm = (float) std::sqrt(sq);
    if (norm > 0.0f) {
        const float inv = 1.0f / norm;
        for (float & x : v) {
            x *= inv;
        }
    }
    return v;
}
