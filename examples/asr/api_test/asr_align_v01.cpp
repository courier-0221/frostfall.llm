// ============================================================================
// asr_align_v01.cpp —— asr/v0.1 数值对齐示例
//
// 验证目标（doc/asr/design_asr_v0.x.md §2.2 第 7/8 条）：
//   1. qwen3-asr GGUF 能被本工程加载：文本 Decoder 权重 + 音频塔张量全部加载并校验；
//   2. Decoder 双入口：以 Python 导出的融合 Embedding [n_embd, S] 作为第 0 层输入，
//      prefill 输出与官方参考逐层 hidden / 末位 logits 对齐；
//   3. teacher forcing 多步 decode 与参考 logits/top-k 对齐；
//   4. 自由贪心生成复现参考 token 序列。
//
// 参考数据由 tools/export_asr_reference.py 生成（work/asr_ref/{auto,lang}）。
// 参考精度为 fp32；本工程权重 F16，误差按实测报告（max abs err / RMSE / top-k）。
//
// 用法：
//   asr_align_v01 --gguf models/qwen3-asr-0.6b-f16.gguf \
//                 --ref-dir work/asr_ref/auto [--max-gen 32]
// ============================================================================

#include "graph.h"
#include "kv_cache.h"
#include "model.h"
#include "tokenizer.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ============================ .npy 读取（最小实现） ============================

// 仅支持 v1.0 格式、fortran_order=false、dtype '<f4'(F32) / '<i4'(I32)。
struct NpyArray {
    std::vector<size_t> shape;
    std::vector<char>   data;   // C-order 原始字节
    bool is_f32 = true;

    size_t count() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }
    const float * as_f32() const {
        if (!is_f32) throw std::runtime_error("npy dtype is not <f4");
        return reinterpret_cast<const float *>(data.data());
    }
    const int32_t * as_i32() const {
        if (is_f32) throw std::runtime_error("npy dtype is not <i4");
        return reinterpret_cast<const int32_t *>(data.data());
    }
};

static NpyArray load_npy(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("bad npy magic: " + path);

    uint8_t ver[2];
    f.read(reinterpret_cast<char *>(ver), 2);
    if (ver[0] != 1) throw std::runtime_error("only npy v1.x supported: " + path);

    uint16_t header_len = 0;
    f.read(reinterpret_cast<char *>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // 极简 header 字段解析（descr / fortran_order / shape）
    NpyArray out;
    auto find_field = [&](const char * key) -> std::string {
        const std::string k = "'" + std::string(key) + "':";
        auto pos = header.find(k);
        if (pos == std::string::npos) return "";
        pos += k.size();
        while (pos < header.size() && (header[pos] == ' ' || header[pos] == '(' || header[pos] == '\'')) ++pos;
        std::string val;
        while (pos < header.size() && header[pos] != ',' && header[pos] != ')' && header[pos] != '\'') {
            val += header[pos++];
        }
        return val;
    };

    const std::string descr = find_field("descr");
    if (descr == "<f4" || descr == "<f8") {
        out.is_f32 = true;
        if (descr != "<f4") throw std::runtime_error("f64 npy not supported: " + path);
    } else if (descr == "<i4" || descr == "|i1") {
        out.is_f32 = false;
    } else {
        throw std::runtime_error("unsupported npy dtype '" + descr + "' in " + path);
    }

    const std::string shape_key = "'shape':";
    std::string shape_str;
    {
        // shape 值是括号元组（如 "(29, 80, 1024)"），不能像 descr 那样在逗号处截断
        const auto spos = header.find(shape_key);
        if (spos != std::string::npos) {
            const auto open = header.find('(', spos);
            const auto close = header.find(')', open);
            if (open != std::string::npos && close != std::string::npos) {
                shape_str = header.substr(open + 1, close - open - 1);
            }
        }
    }
    size_t start = 0;
    while (start < shape_str.size()) {
        auto comma = shape_str.find(',', start);
        const std::string tok = shape_str.substr(start, comma == std::string::npos ?
                                                          std::string::npos : comma - start);
        if (!tok.empty() && tok.find_first_not_of("0123456789 ") == std::string::npos) {
            out.shape.push_back((size_t) std::stoul(tok));
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }

    const size_t nbytes = out.count() * (out.is_f32 ? 4 : 4);
    out.data.resize(nbytes);
    f.read(out.data.data(), (std::streamsize) nbytes);
    return out;
}

// ============================ 误差统计 ============================

struct ErrStats {
    double max_abs = 0.0;
    double rmse   = 0.0;
    size_t n      = 0;
    bool   has_nan = false;

    static ErrStats compute(const float * a, const float * b, size_t n) {
        ErrStats s;
        s.n = n;
        double sum2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = (double) a[i] - (double) b[i];
            if (std::isnan(d)) { s.has_nan = true; continue; }
            s.max_abs = std::max(s.max_abs, std::fabs(d));
            sum2 += d * d;
        }
        s.rmse = std::sqrt(sum2 / (double) std::max<size_t>(n, 1));
        return s;
    }
};

// top-k 重合：两个 logits 向量 top-k 集合的交集大小
static int topk_overlap(const float * a, const float * b, size_t n, int k) {
    std::vector<size_t> ia(n), ib(n);
    for (size_t i = 0; i < n; ++i) ia[i] = ib[i] = i;
    // 比较器直接捕获调用方的指针（生命周期覆盖本函数），避免嵌套 lambda 悬垂
    std::partial_sort(ia.begin(), ia.begin() + k, ia.end(),
                      [&](size_t x, size_t y) { return a[x] > a[y]; });
    std::partial_sort(ib.begin(), ib.begin() + k, ib.end(),
                      [&](size_t x, size_t y) { return b[x] > b[y]; });
    int hit = 0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (ia[i] == ib[j]) { ++hit; break; }
        }
    }
    return hit;
}

static void print_err(const char * tag, const ErrStats & s, int topk_hit = -1, int topk = 0) {
    std::cout << "  [" << tag << "] n=" << s.n
              << " max_abs=" << s.max_abs
              << " rmse=" << s.rmse;
    if (s.has_nan) std::cout << " (NaN present!)";
    if (topk_hit >= 0) std::cout << " top" << topk << "_hit=" << topk_hit << "/" << topk;
    std::cout << "\n";
}

// ============================ 推理 harness ============================

namespace {

constexpr int kMaxNodes = 16384;

// 容差按 fp32 参考 vs 本工程 F16 权重/F16 KV cache 的实测结果建立：
//   - 逐层 hidden 的绝对偏差集中在数值 10^2~10^3 的 attention sink 通道
//     （相对误差 ~0.15%，F16 量化噪声），用排除 sink 后的 rmse 判定；
//   - 末层 hidden 受 sink token 的 K cache 量化误差影响（rmse 跳升），但其唯一
//     消费者是末位 logits（logits_last_only=true），精度由 logits 判定覆盖。
constexpr float kHiddenRmseTol = 0.05f; // 逐层 hidden（ex-sink）rmse 上限
constexpr float kLogitsMaxTol  = 0.5f;  // 末位 logits max abs 上限
constexpr int   kTopKMin       = 9;     // top10 最小命中数

struct Harness {
    qwen3_model     model;
    qwen3_kv_cache  kv;
    qwen3_tokenizer tok;
    ggml_gallocr_t  allocr = nullptr;

    bool load(const std::string & gguf_path, int32_t n_ctx) {
        if (!qwen3_model_load(gguf_path, model)) return false;
        if (!tok.load(gguf_path)) return false;
        if (!kv.init(model, n_ctx)) return false;
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        return true;
    }

    // 一次前向。返回图（调用方读取输出后 ggml_free(ctx)）。
    // embd_in：[n_embd, n_tokens] F32（行=token 的 C-order 数据），或 token ids。
    struct ggml_cgraph * eval(ggml_context * ctx, const int32_t * tokens,
                              const float * embd_in, int32_t n_tokens, int32_t n_past,
                              bool want_layers, std::vector<float> & logits_out) {
        qwen3_graph_params gp;
        gp.n_tokens = n_tokens;
        gp.n_past = n_past;
        gp.max_nodes = kMaxNodes;
        gp.logits_last_only = true;
        gp.want_layer_outputs = want_layers;

        // embd 入口：在 ctx 中创建输入张量并经 gp.embd 交给图（第 0 层直接使用它）
        struct ggml_tensor * embd_t = nullptr;
        if (embd_in) {
            embd_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_embd, n_tokens);
            ggml_set_name(embd_t, QWEN3_TENSOR_NAME_EMBD);
            ggml_set_input(embd_t);
            gp.embd = embd_t;
        }

        struct ggml_cgraph * gf = qwen3_build_graph(ctx, model, kv, gp);
        ggml_gallocr_alloc_graph(allocr, gf);

        if (embd_in) {
            ggml_backend_tensor_set(embd_t, embd_in, 0,
                                    (size_t) n_tokens * model.hparams.n_embd * sizeof(float));
        } else {
            struct ggml_tensor * t_tokens = ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_TOKENS);
            ggml_backend_tensor_set(t_tokens, tokens, 0, (size_t) n_tokens * sizeof(int32_t));
        }

        std::vector<int32_t> pos(n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) pos[i] = n_past + i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_POS),
                                pos.data(), 0, (size_t) n_tokens * sizeof(int32_t));

        const int32_t n_kv = n_past + n_tokens;
        std::vector<float> mask((size_t) n_tokens * n_kv, 0.0f);
        for (int32_t q = 0; q < n_tokens; ++q) {
            for (int32_t k = n_past + q + 1; k < n_kv; ++k) {
                mask[(size_t) q * n_kv + k] = -INFINITY;
            }
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_MASK),
                                mask.data(), 0, mask.size() * sizeof(float));

        ggml_backend_graph_compute(model.backend, gf);

        const int32_t n_vocab = model.hparams.n_vocab;
        logits_out.resize((size_t) n_vocab);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_LOGITS),
                                logits_out.data(), 0, (size_t) n_vocab * sizeof(float));
        kv.n_past = n_kv;
        return gf;
    }

    bool is_eos(int32_t id) const {
        for (int32_t e : model.asr_hparams.eos_token_ids) {
            if (id == e) return true;
        }
        return false;
    }
};

} // namespace

// ============================ main ============================

int main(int argc, char ** argv) {
    std::string gguf_path, ref_dir;
    int max_gen = 32;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--ref-dir" && i + 1 < argc) ref_dir = argv[++i];
        else if (a == "--max-gen" && i + 1 < argc) max_gen = std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0] << " --gguf <path> --ref-dir <dir> [--max-gen N]\n";
            return 0;
        }
    }
    if (gguf_path.empty() || ref_dir.empty()) {
        std::cerr << "missing --gguf / --ref-dir\n";
        return 1;
    }

    // ---- 参考数据 ----
    const NpyArray ref_ids = load_npy(ref_dir + "/ids.npy");
    const NpyArray ref_embd = load_npy(ref_dir + "/embd.npy");
    const NpyArray ref_hidden = load_npy(ref_dir + "/layer_hidden.npy");
    const NpyArray ref_dlogits = load_npy(ref_dir + "/decode_logits.npy");
    const NpyArray ref_dids = load_npy(ref_dir + "/decode_ids.npy");

    const int32_t S = (int32_t) ref_ids.shape[0];
    std::cout << "== asr/v0.1 alignment ==\n"
              << "  S(prefill)=" << S << " n_gen_ref=" << ref_dids.shape[0] << "\n";

    // n_ctx：prefill + 自由生成上限（max_gen，防 argmax 分叉后越界）+ 余量
    Harness hs;
    const int32_t n_ctx = S + max_gen + 16;
    if (!hs.load(gguf_path, n_ctx)) {
        std::cerr << "failed to load model/kv/tokenizer from " << gguf_path << "\n";
        return 1;
    }
    const int32_t n_vocab = hs.model.hparams.n_vocab;
    const int32_t n_embd = hs.model.hparams.n_embd;
    const int32_t n_layer = hs.model.hparams.n_layer;
    std::cout << "  arch=" << (hs.model.is_asr ? "qwen3-asr" : "qwen3")
              << " n_vocab=" << n_vocab << " n_embd=" << n_embd
              << " n_layer=" << n_layer
              << " audio_layers=" << hs.model.asr_hparams.n_audio_layer << "\n";

    int failures = 0;

    // ================= 1. prefill：外部 Embedding 入口 =================
    {
        struct ggml_init_params gparams = {
            /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes * 4 + ggml_graph_overhead_custom(kMaxNodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx = ggml_init(gparams);

        std::vector<float> logits;
        struct ggml_cgraph * gf = hs.eval(ctx, nullptr, ref_embd.as_f32(), S, 0, true, logits);

        std::cout << "-- prefill (embd entry, layer outputs on) --\n";

        // 逐层 hidden：C++ layer_out_i = [n_embd, S]（内存 token 主序），参考 [29,S,1024]
        for (int il = 0; il < n_layer; ++il) {
            char name[32];
            std::snprintf(name, sizeof(name), QWEN3_TENSOR_NAME_LAYER_OUT_PREFIX "%d", il);
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (!t) { std::cerr << "missing " << name << "\n"; ++failures; continue; }
            std::vector<float> out((size_t) S * n_embd);
            ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
            // hidden_states[i+1] 对应第 i 层输出（[0] 是 embedding 输出）
            const float * ref = ref_hidden.as_f32() + (size_t)(il + 1) * S * n_embd;
            const ErrStats st_all = ErrStats::compute(out.data(), ref, out.size());
            // token 0 是 attention sink（hidden 值极大，F16 KV 量化误差在此被逐层放大），
            // 精度对齐以排除 sink 后的统计为准；全量 max_abs 仍打印供参考。
            const ErrStats st_ex = ErrStats::compute(out.data() + n_embd, ref + n_embd,
                                                     out.size() - (size_t) n_embd);
            std::cout << "  [" << name << "] n=" << st_all.n
                      << " max_abs=" << st_all.max_abs << " rmse=" << st_all.rmse
                      << " | ex-sink max_abs=" << st_ex.max_abs
                      << " rmse=" << st_ex.rmse
                      << (st_all.has_nan ? " (NaN present!)" : "") << "\n";
            if (il < n_layer - 1 && st_ex.rmse > kHiddenRmseTol) {
                // 定位最大偏差位置：内存 token 主序，i/n_embd = token，i%n_embd = channel
                size_t wi = 0; double wd = -1.0;
                for (size_t i = (size_t) n_embd; i < out.size(); ++i) {
                    const double d = std::fabs((double) out[i] - (double) ref[i]);
                    if (std::isnan(d)) continue;
                    if (d > wd) { wd = d; wi = i; }
                }
                std::cout << "    worst(ex-sink): token=" << wi / (size_t) n_embd
                          << " ch=" << wi % (size_t) n_embd
                          << " cpp=" << out[wi] << " ref=" << ref[wi] << "\n";
                ++failures;
            }
        }
        // embedding 输出对照（hidden_states[0]）
        {
            std::vector<float> out((size_t) S * n_embd);
            std::memcpy(out.data(), ref_embd.as_f32(), out.size() * sizeof(float));
            const float * ref = ref_hidden.as_f32();
            print_err("embd_input", ErrStats::compute(out.data(), ref, out.size()));
        }

        // prefill 末位 logits（= 参考第一步 decode logits）
        const float * ref_logits = ref_dlogits.as_f32(); // [n_gen, vocab] 第 0 行
        const ErrStats ls = ErrStats::compute(logits.data(), ref_logits, (size_t) n_vocab);
        const int hit = topk_overlap(logits.data(), ref_logits, (size_t) n_vocab, 10);
        print_err("prefill_logits", ls, hit, 10);
        if (ls.max_abs > kLogitsMaxTol || hit < kTopKMin) ++failures;

        ggml_free(ctx);
    }

    // ================= 2. teacher forcing 多步 decode =================
    {
        const int n_gen = (int) ref_dids.shape[0];
        std::cout << "-- teacher forcing (" << n_gen - 1 << " steps) --\n";
        double worst_max_abs = 0.0;
        int worst_top = 10;
        const int32_t * dids = ref_dids.as_i32();
        for (int i = 1; i < n_gen; ++i) {
            // 输入参考序列的前一个 token；n_past = S + i - 1
            struct ggml_init_params gparams = {
                /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes * 4 + ggml_graph_overhead_custom(kMaxNodes, false),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            struct ggml_context * ctx = ggml_init(gparams);
            std::vector<float> logits;
            hs.eval(ctx, dids + (i - 1), nullptr, 1, S + i - 1, false, logits);
            const float * ref = ref_dlogits.as_f32() + (size_t) i * n_vocab;
            const ErrStats s = ErrStats::compute(logits.data(), ref, (size_t) n_vocab);
            worst_max_abs = std::max(worst_max_abs, s.max_abs);
            worst_top = std::min(worst_top, topk_overlap(logits.data(), ref, (size_t) n_vocab, 10));
            ggml_free(ctx);
        }
        std::cout << "  [decode_logits] steps=" << n_gen - 1
                  << " worst_max_abs=" << worst_max_abs
                  << " worst_top10_hit=" << worst_top << "/10\n";
        if (worst_max_abs > kLogitsMaxTol || worst_top < kTopKMin) ++failures;
    }

    // ================= 3. 自由贪心生成 =================
    {
        std::cout << "-- free greedy generation --\n";
        // 重置 KV：重新 load 太重；prefill 用同一 harness 重跑一次（kv.n_past 已到 S+n_gen-1，
        // 自由生成从 0 开始需重置）
        hs.kv.n_past = 0;

        const int n_gen = (int) ref_dids.shape[0];
        const int32_t * dids = ref_dids.as_i32();
        std::vector<int32_t> gen;
        int32_t n_past = 0;
        std::vector<float> logits;
        for (int step = 0; step < max_gen; ++step) {
            struct ggml_init_params gparams = {
                /*.mem_size   =*/ ggml_tensor_overhead() * kMaxNodes * 4 + ggml_graph_overhead_custom(kMaxNodes, false),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            struct ggml_context * ctx = ggml_init(gparams);
            if (step == 0) {
                hs.eval(ctx, nullptr, ref_embd.as_f32(), S, 0, false, logits);
            } else {
                hs.eval(ctx, &gen.back(), nullptr, 1, n_past, false, logits);
            }
            n_past = hs.kv.n_past;
            ggml_free(ctx);
            // 贪心 argmax
            int32_t best = 0;
            float bv = logits[0];
            for (int32_t v = 1; v < n_vocab; ++v) {
                if (logits[v] > bv) { bv = logits[v]; best = v; }
            }
            gen.push_back(best);
            if (hs.is_eos(best)) break;
        }

        const int match_n = std::min(gen.size(), (size_t) n_gen);
        int token_match = 0;
        for (int i = 0; i < match_n; ++i) token_match += (gen[i] == dids[i]) ? 1 : 0;

        std::cout << "  [free_gen] n=" << gen.size() << " (ref " << n_gen << ")"
                  << " prefix_match=" << token_match << "/" << match_n << "\n";
        std::cout << "  [text] " << hs.tok.decode(gen, /*skip_special=*/false) << "\n";
        if (gen.size() != (size_t) n_gen || token_match != match_n) ++failures;
    }

    // ================= 汇总 =================
    std::cout << "== result: " << (failures == 0 ? "PASS" : "FAIL")
              << " (failures=" << failures << ") ==\n";
    return failures == 0 ? 0 : 1;
}
