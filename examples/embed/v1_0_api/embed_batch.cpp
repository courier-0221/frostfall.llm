#include "embedding_engine.h"
#include "embed_test_util.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <embed_conf.json> [--target-dim N]\n", argv[0]);
        return 1;
    }

    int target_dim = 0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--target-dim" && i + 1 < argc) {
            target_dim = std::atoi(argv[++i]);
        }
    }

    Frostfall::EmbeddingEngine engine;
    if (!engine.InitFromJsonFile(argv[1])) {
        return 2;
    }

    std::vector<std::string> queries = {
        "What is the capital of China?",
        "Explain gravity"
    };
    std::vector<std::string> docs = {
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies toward each other."
    };

    Frostfall::EmbedRequest qreq;
    qreq.texts = queries;
    qreq.is_query = true;
    qreq.target_dim = target_dim;
    Frostfall::EmbedResponse qresp = engine.Embed(qreq);
    if (!qresp.error.empty()) {
        std::fprintf(stderr, "query batch failed: %s\n", qresp.error.c_str());
        return 3;
    }

    Frostfall::EmbedRequest dreq;
    dreq.texts = docs;
    dreq.is_query = false;
    dreq.target_dim = target_dim;
    Frostfall::EmbedResponse dresp = engine.Embed(dreq);
    if (!dresp.error.empty()) {
        std::fprintf(stderr, "doc batch failed: %s\n", dresp.error.c_str());
        return 4;
    }

    for (size_t i = 0; i < qresp.embeddings.size(); ++i) {
        std::string name = "query" + std::to_string(i);
        embed_print_vector_summary(name.c_str(), qresp.embeddings[i]);
    }
    for (size_t i = 0; i < dresp.embeddings.size(); ++i) {
        std::string name = "doc" + std::to_string(i);
        embed_print_vector_summary(name.c_str(), dresp.embeddings[i]);
    }

    std::printf("similarity matrix:\n");
    for (const auto& q : qresp.embeddings) {
        std::printf("[");
        for (size_t j = 0; j < dresp.embeddings.size(); ++j) {
            std::printf("%s%.4f", j == 0 ? "" : ", ", embed_dot(q, dresp.embeddings[j]));
        }
        std::printf("]\n");
    }
    return 0;
}
