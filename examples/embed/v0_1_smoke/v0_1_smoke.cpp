// frostfall.embed v0.1 —— 最小 smoke CLI
//
// 用法：
//   ./v0_1_smoke <model.gguf> [--text "..."] [--is-query] [--task "..."] \
//                [--target-dim N] [--threads N] [--out FILE]
//
// 若不传 --text，则从 stdin 一次性读取（直到 EOF）。
//
// stdout 只打印一行 JSON（供对拍脚本 grep）：
//   {"dim": 1024, "n_tokens": 12, "norm": 1.0, "v8": [f0, ..., f7]}
// 传 --out FILE 时把完整向量以 JSON（额外含 "v_all" 字段）写入 FILE，供全量 cosine 对拍。
// 其余日志走 stderr（frostfall 的 LOG(*) 就是写 stderr）。
//
// 关键行为（对齐官方 transformers 用法）：
//   1. Qwen3-Embedding GGUF 元数据 tokenizer.ggml.add_eos_token = True，
//      官方 tokenizer 会在末尾追加 <|endoftext|>(151643)。C++ 侧 encode 后必须手动追加。
//   2. is_query=true 时前置 "Instruct: {task}\nQuery:" 前缀；document 侧不加前缀。
//   3. 输出向量按 last-token pool -> MRL 截断 -> L2 归一化的顺序（详见 pooling.cpp）。

#include "embed_graph.h"
#include "log.h"
#include "model.h"
#include "pooling.h"
#include "qwen3_embed_prompt.h"
#include "tokenizer.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s <model.gguf> [--text \"...\"] [--is-query] [--task \"...\"] "
        "[--target-dim N] [--threads N] [--out FILE]\n"
        "  --text        input text; if omitted, read from stdin until EOF\n"
        "  --is-query    prepend \"Instruct: {task}\\nQuery:\" to input\n"
        "  --task        task description (default = official web-search prompt)\n"
        "  --target-dim  MRL truncation dim in [32, 1024]; 0 or 1024 = full\n"
        "  --threads     CPU threads for backend (default 8)\n"
        "  --out FILE    write full JSON (incl. v_all) to FILE for diff\n",
        prog);
}

std::string read_stdin_all() {
    std::stringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string text;
    bool        has_text   = false;
    bool        is_query   = false;
    std::string task;
    int32_t     target_dim = 0;
    int32_t     n_threads  = 8;
    std::string out_path;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--text" && i + 1 < argc) {
            text = argv[++i]; has_text = true;
        } else if (a == "--is-query") {
            is_query = true;
        } else if (a == "--task" && i + 1 < argc) {
            task = argv[++i];
        } else if (a == "--target-dim" && i + 1 < argc) {
            target_dim = std::atoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            LOG(ERROR) << "unknown arg: " << a;
            print_usage(argv[0]);
            return 1;
        }
    }
    if (!has_text) {
        text = read_stdin_all();
        // 剥去 stdin 结尾的换行（避免与官方 transformers 输入差异）。
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
    }

    // ---- 1. 加载模型 + tokenizer ----
    qwen3_model model;
    if (!qwen3_model_load(model_path, model)) {
        LOG(ERROR) << "failed to load model: " << model_path;
        return 2;
    }
    if (n_threads > 0) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    qwen3_tokenizer tokenizer;
    if (!tokenizer.load(model_path)) {
        LOG(ERROR) << "failed to load tokenizer: " << model_path;
        return 3;
    }

    // ---- 2. instruction 模板拼接 ----
    const std::string input = is_query
        ? qwen3_embed_build_query_text(text, task)
        : text;

    // ---- 3. encode + 手动追加 eos (add_eos_token=True 官方规则) ----
    std::vector<int32_t> ids = tokenizer.encode(input);
    // Qwen3-Embedding 的 eos 是 <|endoftext|> = 151643。
    // 从 hparams 里读，兜底 151643。
    int32_t eos_id = model.hparams.eos_token_id > 0 ? model.hparams.eos_token_id : 151643;
    ids.push_back(eos_id);

    const int32_t n_tokens = (int32_t) ids.size();
    if (n_tokens <= 0) {
        LOG(ERROR) << "empty input after tokenization";
        return 4;
    }
    LOG(INFO) << "input tokens: " << n_tokens << " (last=" << ids.back() << ")";

    // ---- 4. 构图 + 分配 + 运行 ----
    constexpr int kMaxNodes = 8192;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes
                          + ggml_graph_overhead_custom(kMaxNodes, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(gparams);

    struct ggml_cgraph * gf = qwen3_embed_build_graph(ctx, model, n_tokens, kMaxNodes);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // 填 tokens
    struct ggml_tensor * t_tokens = ggml_graph_get_tensor(gf, QWEN3_EMBED_TENSOR_NAME_TOKENS);
    ggml_backend_tensor_set(t_tokens, ids.data(), 0, n_tokens * sizeof(int32_t));

    // 填 positions 0..n-1
    std::vector<int32_t> pos_buf(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) pos_buf[i] = i;
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, QWEN3_EMBED_TENSOR_NAME_POS);
    ggml_backend_tensor_set(t_pos, pos_buf.data(), 0, n_tokens * sizeof(int32_t));

    // 因果 mask：ne0 = key (n_tokens)，ne1 = query (n_tokens)；k > q 处填 -inf。
    std::vector<float> mask_buf((size_t) n_tokens * n_tokens, 0.0f);
    for (int32_t q = 0; q < n_tokens; ++q) {
        for (int32_t k = q + 1; k < n_tokens; ++k) {
            mask_buf[(size_t) q * n_tokens + k] = -INFINITY;
        }
    }
    struct ggml_tensor * t_mask = ggml_graph_get_tensor(gf, QWEN3_EMBED_TENSOR_NAME_MASK);
    ggml_backend_tensor_set(t_mask, mask_buf.data(), 0, mask_buf.size() * sizeof(float));

    ggml_backend_graph_compute(model.backend, gf);

    // ---- 5. 取隐层输出 + pooling ----
    struct ggml_tensor * t_hidden = ggml_graph_get_tensor(gf, QWEN3_EMBED_TENSOR_NAME_HIDDEN);
    const int32_t n_embd = model.hparams.n_embd;

    std::vector<float> hidden((size_t) n_embd * n_tokens);
    ggml_backend_tensor_get(t_hidden, hidden.data(), 0, hidden.size() * sizeof(float));

    std::vector<float> emb = qwen3_embed_pool_and_normalize(hidden.data(), n_embd, n_tokens, target_dim);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);

    // ---- 6. 打印结果（stdout 一行 JSON）----
    double norm2 = 0.0;
    for (float x : emb) norm2 += (double) x * (double) x;
    const int32_t out_dim = (int32_t) emb.size();
    const int32_t k = std::min(8, out_dim);

    std::printf("{\"dim\": %d, \"n_tokens\": %d, \"norm\": %.6f, \"v8\": [",
                out_dim, n_tokens, std::sqrt(norm2));
    for (int32_t i = 0; i < k; ++i) {
        std::printf("%s%.6f", i == 0 ? "" : ", ", emb[i]);
    }
    std::printf("]}\n");

    if (!out_path.empty()) {
        std::FILE * fp = std::fopen(out_path.c_str(), "w");
        if (!fp) {
            LOG(ERROR) << "cannot open --out file: " << out_path;
            return 5;
        }
        std::fprintf(fp, "{\"dim\": %d, \"n_tokens\": %d, \"norm\": %.9f, \"v_all\": [",
                     out_dim, n_tokens, std::sqrt(norm2));
        for (int32_t i = 0; i < out_dim; ++i) {
            std::fprintf(fp, "%s%.9f", i == 0 ? "" : ", ", emb[i]);
        }
        std::fprintf(fp, "]}\n");
        std::fclose(fp);
    }
    return 0;
}
