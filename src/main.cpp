// frostfall v0.2 —— 自研分词器 + 增量 KV cache + 计时统计。
//
// 主链路：加载 GGUF -> 构建 tokenizer -> 文本 encode -> prefill(一次性整段) ->
//         decode(每步 1 个 token，K/V 增量追加) -> argmax 贪心 -> 流式解码打印。
//
// 相比 v0.1 的变化：
//   1. 分词自己做（src/tokenizer.*），不再依赖 Python 预处理；用 -p 直接传文本。
//   2. 增量 KV cache（src/kv_cache.*），复杂度从 O(n^2) 降到 O(n)。
//   3. 打印各阶段耗时、tokens/s，以及权重 / KV cache 内存占用。
//   4. 仍保留 -i/-o（token id 文件）以兼容对拍脚本 scripts/compare_llamacpp.py。

#include "common.h"
#include "graph.h"
#include "kv_cache.h"
#include "model.h"
#include "tokenizer.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "log.h"

#include <cmath>
#include <cstdint>
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
    std::string prompt;               // -p：直接传文本（自研分词）
    std::string tokens_in;            // -i：读 token id 文件（兼容对拍脚本）
    std::string tokens_out;           // -o：写完整 token id 序列（兼容对拍脚本）
    int32_t     n_predict    = 64;
    int32_t     n_threads    = 4;
    int32_t     n_ctx        = 4096;  // KV cache 上下文上限
    bool        no_chat      = false; // 不套 chat 模板，直接对纯文本编码
    bool        enable_think = false; // Qwen3 chat 模板是否启用思考模式
    bool        has_prompt   = false;
};

void print_usage(const char * prog) {
    fprintf(stderr,
        "frostfall v0.2 - 基于 ggml 的 Qwen3-0.6B 教学版推理框架（自研分词 + 增量 KV cache）\n"
        "\n"
        "usage: %s -m <model.gguf> (-p <prompt> | -i <tokens_in.txt>) [options]\n"
        "\n"
        "  -m, --model        FILE  Qwen3 GGUF 模型路径（必填）\n"
        "  -p, --prompt       TEXT  用户输入文本（自研分词器编码；默认套 Qwen3 chat 模板）\n"
        "  -i, --tokens-in    FILE  改为读入空白分隔的 token id（兼容对拍脚本，与 -p 二选一）\n"
        "  -o, --tokens-out   FILE  把完整 token id 序列（prompt+生成）写入此文件（可选）\n"
        "  -n, --n-predict    N     最多生成多少个新 token（默认 64）\n"
        "  -c, --ctx-size     N     KV cache 上下文上限（默认 4096）\n"
        "  -t, --threads      N     CPU 计算线程数（默认 4）\n"
        "      --no-chat-template   不套 chat 模板，直接对 -p 的纯文本编码\n"
        "      --think              启用 Qwen3 思考模式（默认关闭）\n",
        prog);
}

bool parse_args(int argc, char ** argv, cli_args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { LOG(ERROR) << "missing value for " << name; exit(1); }
            return argv[++i];
        };
        if      (arg == "-m" || arg == "--model")      args.model_path = next_value(arg.c_str());
        else if (arg == "-p" || arg == "--prompt")     { args.prompt = next_value(arg.c_str()); args.has_prompt = true; }
        else if (arg == "-i" || arg == "--tokens-in")  args.tokens_in = next_value(arg.c_str());
        else if (arg == "-o" || arg == "--tokens-out") args.tokens_out = next_value(arg.c_str());
        else if (arg == "-n" || arg == "--n-predict")  args.n_predict = std::stoi(next_value(arg.c_str()));
        else if (arg == "-c" || arg == "--ctx-size")   args.n_ctx = std::stoi(next_value(arg.c_str()));
        else if (arg == "-t" || arg == "--threads")    args.n_threads = std::stoi(next_value(arg.c_str()));
        else if (arg == "--no-chat-template")          args.no_chat = true;
        else if (arg == "--think")                     args.enable_think = true;
        else if (arg == "-h" || arg == "--help")       { print_usage(argv[0]); exit(0); }
        else { LOG(ERROR) << "unknown argument '" << arg << "'"; return false; }
    }
    if (args.model_path.empty() || (!args.has_prompt && args.tokens_in.empty())) {
        LOG(ERROR) << "-m/--model and one of -p/--prompt or -i/--tokens-in are required";
        return false;
    }
    return true;
}

std::vector<int32_t> read_token_ids(const std::string & path) {
    std::ifstream fin(path);
    std::vector<int32_t> ids;
    int32_t v;
    while (fin >> v) ids.push_back(v);
    return ids;
}

int32_t argmax(const float * logits, int32_t n) {
    int32_t best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n; ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

} // namespace

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) { print_usage(argv[0]); return 1; }

    // ---- 1. 加载模型 ----
    ff::timer t_total;
    ff::timer t_stage;

    qwen3_model model;
    if (!qwen3_model_load(args.model_path, model)) {
        LOG(ERROR) << "failed to load model from '" << args.model_path << "'";
        return 1;
    }
    const double t_load_ms = t_stage.elapsed_ms();
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, args.n_threads);
    }

    // ---- 2. 构建 tokenizer 并编码 prompt ----
    qwen3_tokenizer tokenizer;
    std::vector<int32_t> ids;
    if (args.has_prompt) {
        if (!tokenizer.load(args.model_path)) {
            LOG(ERROR) << "failed to load tokenizer from '" << args.model_path << "'";
            return 1;
        }
        const std::string text = args.no_chat ? args.prompt
                                              : ff::apply_chat_template(args.prompt, args.enable_think);
        ids = tokenizer.encode(text);
    } else {
        // 兼容模式：直接读入 token id（供对拍脚本使用，不做分词）
        ids = read_token_ids(args.tokens_in);
    }
    if (ids.empty()) { LOG(ERROR) << "empty prompt token sequence"; return 1; }

    const int32_t eos_id = tokenizer.eos_id >= 0 ? tokenizer.eos_id : model.hparams.eos_token_id;
    LOG(INFO) << "prompt has " << ids.size() << " tokens";

    // ---- 3. 初始化 KV cache ----
    const int32_t n_prompt = (int32_t) ids.size();
    int32_t n_ctx = args.n_ctx;
    if (n_ctx > model.hparams.n_ctx_train) n_ctx = model.hparams.n_ctx_train;
    if (n_prompt + args.n_predict > n_ctx) {
        LOG(WARNING) << "prompt(" << n_prompt << ") + n_predict(" << args.n_predict
                     << ") exceeds ctx-size(" << n_ctx << "), generation will be truncated";
    }

    qwen3_kv_cache kv;
    if (!kv.init(model, n_ctx)) return 1;

    const int32_t n_vocab   = model.hparams.n_vocab;
    const int     max_nodes = 8192;
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    std::vector<float>   mask_buf;
    std::vector<int32_t> pos_buf;

    // 前向一个 batch：把 [batch(n 个)] 写入 cache 的 [n_past, n_past+n)，返回“最后一个位置”的 argmax。
    auto eval_batch = [&](const int32_t * batch, int32_t n, int32_t n_past) -> int32_t {
        const int32_t n_kv = n_past + n;

        struct ggml_init_params gparams = {
            /*.mem_size   =*/ ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx = ggml_init(gparams);

        struct ggml_cgraph * gf = qwen3_build_graph(ctx, model, kv, n, n_past, max_nodes);
        ggml_gallocr_alloc_graph(allocr, gf);

        struct ggml_tensor * t_tokens = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_TOKENS);
        ggml_backend_tensor_set(t_tokens, batch, 0, n * sizeof(int32_t));

        pos_buf.resize(n);
        for (int32_t i = 0; i < n; ++i) pos_buf[i] = n_past + i;
        struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_POS);
        ggml_backend_tensor_set(t_pos, pos_buf.data(), 0, n * sizeof(int32_t));

        // 因果 mask：ne0=k(0..n_kv-1)，ne1=q(0..n-1)；query 的绝对位置是 n_past+q，可见 k<=n_past+q。
        mask_buf.assign((size_t) n * n_kv, 0.0f);
        for (int32_t q = 0; q < n; ++q) {
            for (int32_t k = n_past + q + 1; k < n_kv; ++k) {
                mask_buf[(size_t) q * n_kv + k] = -INFINITY;
            }
        }
        struct ggml_tensor * t_mask = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_MASK);
        ggml_backend_tensor_set(t_mask, mask_buf.data(), 0, mask_buf.size() * sizeof(float));

        ggml_backend_graph_compute(model.backend, gf);

        struct ggml_tensor * t_logits = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_LOGITS);
        std::vector<float> logits(n_vocab);
        ggml_backend_tensor_get(t_logits, logits.data(),
                (size_t)(n - 1) * n_vocab * sizeof(float), n_vocab * sizeof(float));

        ggml_free(ctx);
        kv.n_past = n_kv; // cache 里现在有 n_kv 个有效 token
        return argmax(logits.data(), n_vocab);
    };

    // ---- 4. prefill：一次性前向整段 prompt ----
    t_stage.reset();
    int32_t next_id = eval_batch(ids.data(), n_prompt, 0);
    const double t_prefill_ms = t_stage.elapsed_ms();

    // ---- 5. decode：每步只前向 1 个新 token ----
    std::string gen_text;
    ids.push_back(next_id);
    {
        const std::string piece = tokenizer.id_to_piece(next_id);
        fputs(piece.c_str(), stdout); // 流式打印
        fflush(stdout);
        gen_text += piece;
    }

    t_stage.reset();
    int32_t n_decoded = 1;
    for (int32_t step = 1; step < args.n_predict; ++step) {
        if (kv.n_past >= n_ctx) { LOG(WARNING) << "reached ctx-size (" << n_ctx << "), stopping"; break; }
        if (next_id == eos_id)  break;

        next_id = eval_batch(&next_id, 1, kv.n_past);
        ids.push_back(next_id);
        ++n_decoded;

        if (next_id == eos_id) break;

        const std::string piece = tokenizer.id_to_piece(next_id);
        fputs(piece.c_str(), stdout);
        fflush(stdout);
        gen_text += piece;
    }
    fputc('\n', stdout);
    const double t_decode_ms = t_stage.elapsed_ms();

    ggml_gallocr_free(allocr);

    // ---- 6. 统计 ----
    const double dec_tps = (n_decoded > 1 && t_decode_ms > 0) ? (n_decoded - 1) * 1000.0 / t_decode_ms : 0.0;
    LOG(INFO) << "timings: load=" << t_load_ms << " ms"
              << ", prefill=" << t_prefill_ms << " ms (" << n_prompt << " tok, "
              << (n_prompt * 1000.0 / t_prefill_ms) << " tok/s)"
              << ", decode=" << t_decode_ms << " ms (" << n_decoded << " tok, "
              << dec_tps << " tok/s)"
              << ", total=" << t_total.elapsed_ms() << " ms";
    LOG(INFO) << "memory: weights=" << ff::format_mb(ggml_backend_buffer_get_size(model.buffer))
              << ", kv_cache=" << ff::format_mb(kv.size_bytes());

    // ---- 7. 可选：把完整 token id 序列写文件（兼容对拍脚本）----
    if (!args.tokens_out.empty()) {
        std::ostringstream oss;
        for (size_t i = 0; i < ids.size(); ++i) { if (i) oss << ' '; oss << ids[i]; }
        std::ofstream fout(args.tokens_out);
        fout << oss.str() << "\n";
    }

    return 0;
}
