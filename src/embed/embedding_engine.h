#pragma once

#include "embed_types.h"

#include <memory>
#include <string>
#include <vector>

namespace Frostfall {

class EmbeddingEngine {
public:
    struct EngineConfig {
        std::string model_path;
        int n_ctx        = 8192;
        int n_threads    = 8;
        int max_batch    = 8;
        int target_dim   = 1024;
        int pad_token_id = 151643;
        std::string default_task =
            "Given a web search query, retrieve relevant passages that answer the query";
    };

    EmbeddingEngine();
    ~EmbeddingEngine();

    EmbeddingEngine(const EmbeddingEngine&) = delete;
    EmbeddingEngine& operator=(const EmbeddingEngine&) = delete;

    bool Init(const EngineConfig& cfg);
    bool InitFromJsonFile(const std::string& path);
    bool InitFromJsonString(const std::string& json);

    EmbedResponse Embed(const EmbedRequest& req);
    std::vector<float> Embed(const std::string& text,
                             bool is_query = false,
                             const std::string& task = "");

    EmbedStats GetLastStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace Frostfall
