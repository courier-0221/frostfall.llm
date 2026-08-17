#pragma once

#include <string>
#include <vector>

namespace Frostfall {

// Embedding 推理阶段统计。load_ms 来自最近一次 Init，其余字段来自最近一次 Embed。
struct EmbedStats {
    double load_ms     = 0.0;
    double tokenize_ms = 0.0;
    double compute_ms  = 0.0;
    double pool_ms     = 0.0;
    int    n_tokens_max = 0;
    int    n_texts      = 0;
};

// 一次 embedding 请求。texts 内的文本会作为一个静态 batch 同图计算。
struct EmbedRequest {
    std::vector<std::string> texts;
    bool        is_query   = false;
    std::string task;
    int         target_dim = 0;
};

// 一次 embedding 响应。error 非空表示失败，embeddings 为空。
struct EmbedResponse {
    std::vector<std::vector<float>> embeddings;
    int         dim = 0;
    EmbedStats  stats;
    std::string error;
};

}  // namespace Frostfall
