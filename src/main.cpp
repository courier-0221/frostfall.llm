// frostfall v0.1 —— 主体链路：加载 GGUF -> 构图 -> 前向 -> argmax 贪心解码 -> 拼接 -> 循环。
//
// v0.1 的两个简化（详见 doc/design.md §5）：
//   1. tokenizer 由 Python 预处理（scripts/encode_prompt.py），本程序只读/写 token id。
//   2. 没有增量 KV cache，每一步都用当前完整 token 序列重新构图、重新计算一遍。

#include "graph.h"
#include "model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct cli_args {
    std::string model_path;
    std::string tokens_in;
    std::string tokens_out;
    int32_t     n_predict = 64;
    int32_t     n_threads = 4;
};

void print_usage(const char * prog) {
    fprintf(stderr,
        "frostfall v0.1 - 基于 ggml 的 Qwen3-0.6B 教学版推理框架\n"
        "\n"
        "usage: %s -m <model.gguf> -i <tokens_in.txt> [-o <tokens_out.txt>] [-n n_predict] [-t n_threads]\n"
        "\n"
        "  -m, --model       FILE   Qwen3 GGUF 模型路径（必填）\n"
        "  -i, --tokens-in   FILE   prompt 的 token id 文件，空白分隔的整数（必填，见 scripts/encode_prompt.py）\n"
        "  -o, --tokens-out  FILE   生成结果（完整 token id 序列）写入此文件（可选）\n"
        "  -n, --n-predict   N      最多生成多少个新 token（默认 64）\n"
        "  -t, --threads     N      CPU 计算线程数（默认 4）\n",
        prog);
}

bool parse_args(int argc, char ** argv, cli_args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto next_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", name);
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model") {
            args.model_path = next_value(arg.c_str());
        } else if (arg == "-i" || arg == "--tokens-in") {
            args.tokens_in = next_value(arg.c_str());
        } else if (arg == "-o" || arg == "--tokens-out") {
            args.tokens_out = next_value(arg.c_str());
        } else if (arg == "-n" || arg == "--n-predict") {
            args.n_predict = std::stoi(next_value(arg.c_str()));
        } else if (arg == "-t" || arg == "--threads") {
            args.n_threads = std::stoi(next_value(arg.c_str()));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    if (args.model_path.empty() || args.tokens_in.empty()) {
        fprintf(stderr, "error: -m/--model and -i/--tokens-in are required\n");
        return false;
    }
    return true;
}

std::vector<int32_t> read_token_ids(const std::string & path) {
    std::ifstream fin(path);
    std::vector<int32_t> ids;
    int32_t v;
    while (fin >> v) {
        ids.push_back(v);
    }
    return ids;
}

int32_t argmax(const float * logits, int32_t n_vocab) {
    int32_t best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

} // namespace

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    const auto t_start = std::chrono::steady_clock::now();

    qwen3_model model;
    if (!qwen3_model_load(args.model_path, model)) {
        fprintf(stderr, "error: failed to load model from '%s'\n", args.model_path.c_str());
        return 1;
    }

    const auto t_loaded = std::chrono::steady_clock::now();
    fprintf(stderr, "main: model loaded in %.2f s\n",
            std::chrono::duration<double>(t_loaded - t_start).count());

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, args.n_threads);
    }

    std::vector<int32_t> ids = read_token_ids(args.tokens_in);
    if (ids.empty()) {
        fprintf(stderr, "error: no token ids read from '%s'\n", args.tokens_in.c_str());
        return 1;
    }
    fprintf(stderr, "main: prompt has %zu tokens\n", ids.size());

    const int32_t n_vocab   = model.hparams.n_vocab;
    const int32_t n_ctx_max = model.hparams.n_ctx_train;
    const int     max_nodes = 8192; // 28 层 Qwen3 每层约 30 个算子节点，留足余量

    // v0.1 没有增量 KV cache：每一步都用当前完整长度重新构图，gallocr 负责按需重新分配中间张量。
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    std::vector<float>   mask_buf;
    std::vector<int32_t> pos_buf;

    for (int32_t step = 0; step < args.n_predict; ++step) {
        if ((int32_t) ids.size() >= n_ctx_max) {
            fprintf(stderr, "main: reached n_ctx_train (%d), stopping\n", n_ctx_max);
            break;
        }

        const int32_t n_tokens = (int32_t) ids.size();

        struct ggml_init_params gparams = {
            /*.mem_size   =*/ ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true, // 张量数据由 gallocr 之后分配，这里只放图结构的元数据
        };
        struct ggml_context * ctx = ggml_init(gparams);

        struct ggml_cgraph * gf = qwen3_build_graph(ctx, model, n_tokens, max_nodes);
        ggml_gallocr_alloc_graph(allocr, gf);

        // ---- 填输入 ----
        struct ggml_tensor * t_tokens = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_TOKENS);
        ggml_backend_tensor_set(t_tokens, ids.data(), 0, n_tokens * sizeof(int32_t));

        pos_buf.resize(n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) {
            pos_buf[i] = i;
        }
        struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_POS);
        ggml_backend_tensor_set(t_pos, pos_buf.data(), 0, n_tokens * sizeof(int32_t));

        // 因果 mask：ne0=k（kv 位置），ne1=q（query 位置），内存里 q 是慢维、k 是快维
        mask_buf.assign((size_t) n_tokens * n_tokens, 0.0f);
        for (int32_t q = 0; q < n_tokens; ++q) {
            for (int32_t k = q + 1; k < n_tokens; ++k) {
                mask_buf[(size_t) q * n_tokens + k] = -INFINITY;
            }
        }
        struct ggml_tensor * t_mask = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_MASK);
        ggml_backend_tensor_set(t_mask, mask_buf.data(), 0, mask_buf.size() * sizeof(float));

        // ---- 计算 ----
        ggml_backend_graph_compute(model.backend, gf);

        // ---- 只取最后一个 token 位置的 logits，贪心 argmax ----
        struct ggml_tensor * t_logits = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_LOGITS);
        std::vector<float> logits(n_vocab);
        ggml_backend_tensor_get(t_logits, logits.data(),
                (size_t)(n_tokens - 1) * n_vocab * sizeof(float), n_vocab * sizeof(float));

        ggml_free(ctx);

        const int32_t next_id = argmax(logits.data(), n_vocab);
        ids.push_back(next_id);

        fprintf(stderr, "main: step %3d -> token %d\n", step, next_id);

        if (model.hparams.eos_token_id >= 0 && next_id == model.hparams.eos_token_id) {
            fprintf(stderr, "main: eos token generated, stopping\n");
            break;
        }
    }

    ggml_gallocr_free(allocr);

    const auto t_end = std::chrono::steady_clock::now();
    fprintf(stderr, "main: generation done in %.2f s, %zu tokens in total\n",
            std::chrono::duration<double>(t_end - t_loaded).count(), ids.size());

    // ---- 输出：完整 token id 序列（prompt + 生成），空白分隔 ----
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) {
            oss << ' ';
        }
        oss << ids[i];
    }
    const std::string out = oss.str();

    printf("%s\n", out.c_str());

    if (!args.tokens_out.empty()) {
        std::ofstream fout(args.tokens_out);
        fout << out << "\n";
    }

    return 0;
}
