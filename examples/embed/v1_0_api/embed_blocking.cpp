#include "embedding_engine.h"
#include "embed_test_util.h"

#include <cstdlib>
#include <cstdio>
#include <string>

namespace {

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <embed_conf.json> [--text \"...\"] [--is-query] [--task \"...\"] [--target-dim N]\n",
        prog);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string conf_path = argv[1];
    std::string text = "The capital of China is Beijing.";
    bool is_query = false;
    std::string task;
    int target_dim = 0;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--text" && i + 1 < argc) {
            text = argv[++i];
        } else if (a == "--is-query") {
            is_query = true;
        } else if (a == "--task" && i + 1 < argc) {
            task = argv[++i];
        } else if (a == "--target-dim" && i + 1 < argc) {
            target_dim = std::atoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    Frostfall::EmbeddingEngine engine;
    if (!engine.InitFromJsonFile(conf_path)) {
        return 2;
    }

    Frostfall::EmbedRequest req;
    req.texts.push_back(text);
    req.is_query = is_query;
    req.task = task;
    req.target_dim = target_dim;

    Frostfall::EmbedResponse resp = engine.Embed(req);
    if (!resp.error.empty()) {
        std::fprintf(stderr, "embed failed: %s\n", resp.error.c_str());
        return 3;
    }

    embed_print_vector_summary("embedding", resp.embeddings.front());
    std::printf("stats: load=%.2fms tokenize=%.2fms compute=%.2fms pool=%.2fms n_texts=%d max_tokens=%d\n",
                resp.stats.load_ms, resp.stats.tokenize_ms, resp.stats.compute_ms,
                resp.stats.pool_ms, resp.stats.n_texts, resp.stats.n_tokens_max);
    return 0;
}
