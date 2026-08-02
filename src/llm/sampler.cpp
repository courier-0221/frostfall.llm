#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// 候选 token：id + 当前 logit + 归一化后的概率（softmax 后填充）。
struct candidate {
    int32_t id;
    float   logit;
    float   p;
};

int32_t argmax(const float * logits, int32_t n) {
    int32_t best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n; ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

} // namespace

void sampler::init(const sampler_params & p) {
    params = p;
    if (params.seed == SAMPLER_SEED_RANDOM) {
        std::random_device rd;
        params.seed = rd();
    }
    rng.seed(params.seed);
}

int32_t sampler::sample(float * logits, int32_t n_vocab, const std::vector<int32_t> & prev) {
    // ---- 1. 重复惩罚：对最近 repeat_last_n 个 token 施加惩罚（llama.cpp 同款做法）----
    if (params.repeat_penalty != 1.0f && !prev.empty()) {
        const int32_t n = (int32_t) prev.size();
        const int32_t last_n = params.repeat_last_n < 0 ? n : std::min(n, params.repeat_last_n);
        for (int32_t i = n - last_n; i < n; ++i) {
            const int32_t t = prev[i];
            if (t < 0 || t >= n_vocab) continue;
            // logit 为正时缩小、为负时放大，都会降低该 token 被选中的概率。
            if (logits[t] > 0.0f) logits[t] /= params.repeat_penalty;
            else                  logits[t] *= params.repeat_penalty;
        }
    }

    // ---- 2. 贪心：temp<=0 时直接 argmax（与 v0.2 一致）----
    if (params.temp <= 0.0f) {
        return argmax(logits, n_vocab);
    }

    // ---- 3. 构造候选并按 logit 降序（top-k 截断）----
    const int32_t k = params.top_k > 0 ? std::min(params.top_k, n_vocab) : n_vocab;

    std::vector<candidate> cands(n_vocab);
    for (int32_t i = 0; i < n_vocab; ++i) {
        cands[i] = { i, logits[i], 0.0f };
    }
    std::partial_sort(cands.begin(), cands.begin() + k, cands.end(),
                      [](const candidate & a, const candidate & b) { return a.logit > b.logit; });
    cands.resize(k);

    // ---- 4. temperature + softmax（数值稳定：减去最大值）----
    const float max_logit = cands.front().logit; // 已排序，front 即最大
    double sum = 0.0;
    for (auto & c : cands) {
        c.p = std::exp((c.logit - max_logit) / params.temp);
        sum += c.p;
    }
    for (auto & c : cands) {
        c.p = (float) (c.p / sum);
    }

    // ---- 5. top-p（nucleus）：保留累计概率首次达到 top_p 的最小候选集 ----
    if (params.top_p < 1.0f) {
        double cum = 0.0;
        size_t keep = cands.size();
        for (size_t i = 0; i < cands.size(); ++i) {
            cum += cands[i].p;
            if (cum >= params.top_p) { keep = i + 1; break; }
        }
        cands.resize(keep);
        // 重新归一化，使截断后的概率和为 1。
        double s = 0.0;
        for (auto & c : cands) s += c.p;
        for (auto & c : cands) c.p = (float) (c.p / s);
    }

    // ---- 6. 按概率随机采样 ----
    std::vector<double> probs;
    probs.reserve(cands.size());
    for (auto & c : cands) probs.push_back(c.p);
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return cands[dist(rng)].id;
}
