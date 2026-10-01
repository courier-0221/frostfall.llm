// ============================================================================
// asr_align_v02.cpp —— asr/v0.2 数值对齐 + 音频塔闭环示例
//
// 相比 v0.1（输入参考融合 Embedding，只验证 Decoder），v0.2 输入参考 Mel，
// 在 C++ 中完成：音频 Encoder（ggml 音频塔）→ ASR Prompt 构造 + 分词 →
// audio_pad 位置特征替换（融合）→ 混合 Embedding prefill → 增量 decode → 文本解析。
//
// 验证目标（doc/asr/design_asr.md §3.2/§3.4）：
//   1. 音频塔逐段对齐：CNN 输出（conv_out）、18 层音频 Transformer 逐层、投影 features；
//   2. ASR Prompt 构造 + 分词与参考 ids 完全一致；
//   3. 融合 Embedding（替换 audio_pad）与参考 embd 对齐；
//   4. prefill 末位 logits / teacher forcing / 自由贪心生成与参考一致，输出转写文本。
//
// 参考数据由 scripts/asr/export_asr_reference.py 生成（work/asr_ref_wav/{auto,lang}）。
// 参考精度 fp32；本工程音频塔与 Decoder 权重均 F16，误差按实测报告。
//
// 用法：
//   asr_align_v02 --gguf models/qwen3-asr-0.6b-f16.gguf \
//                 --ref-dir work/asr_ref_wav/auto --mode auto [--max-gen 32]
//   asr_align_v02 ... --mode lang --language Chinese
// ============================================================================

#include "audio_encoder.h"
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

// ============================ .npy 读取（最小实现，同 v0.1） ============================

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<char>   data;
    bool is_f32 = true;

    size_t count() const { size_t n = 1; for (size_t d : shape) n *= d; return n; }
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
    char magic[6]; f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("bad npy magic: " + path);
    uint8_t ver[2]; f.read(reinterpret_cast<char *>(ver), 2);
    if (ver[0] != 1) throw std::runtime_error("only npy v1.x supported: " + path);
    uint16_t header_len = 0; f.read(reinterpret_cast<char *>(&header_len), 2);
    std::string header(header_len, '\0'); f.read(header.data(), header_len);

    NpyArray out;
    auto find_field = [&](const char * key) -> std::string {
        const std::string k = "'" + std::string(key) + "':";
        auto pos = header.find(k);
        if (pos == std::string::npos) return "";
        pos += k.size();
        while (pos < header.size() && (header[pos] == ' ' || header[pos] == '(' || header[pos] == '\'')) ++pos;
        std::string val;
        while (pos < header.size() && header[pos] != ',' && header[pos] != ')' && header[pos] != '\'') val += header[pos++];
        return val;
    };
    const std::string descr = find_field("descr");
    if (descr == "<f4") out.is_f32 = true;
    else if (descr == "<i4" || descr == "|i1") out.is_f32 = false;
    else throw std::runtime_error("unsupported npy dtype '" + descr + "' in " + path);

    std::string shape_str;
    {
        const auto spos = header.find("'shape':");
        if (spos != std::string::npos) {
            const auto open = header.find('(', spos);
            const auto close = header.find(')', open);
            if (open != std::string::npos && close != std::string::npos)
                shape_str = header.substr(open + 1, close - open - 1);
        }
    }
    size_t start = 0;
    while (start < shape_str.size()) {
        auto comma = shape_str.find(',', start);
        const std::string tok = shape_str.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty() && tok.find_first_not_of("0123456789 ") == std::string::npos)
            out.shape.push_back((size_t) std::stoul(tok));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    const size_t nbytes = out.count() * 4;
    out.data.resize(nbytes);
    f.read(out.data.data(), (std::streamsize) nbytes);
    return out;
}

// ============================ 误差统计 ============================

struct ErrStats {
    double max_abs = 0.0, rmse = 0.0; size_t n = 0; bool has_nan = false;
    static ErrStats compute(const float * a, const float * b, size_t n) {
        ErrStats s; s.n = n; double sum2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = (double) a[i] - (double) b[i];
            if (std::isnan(d)) { s.has_nan = true; continue; }
            s.max_abs = std::max(s.max_abs, std::fabs(d)); sum2 += d * d;
        }
        s.rmse = std::sqrt(sum2 / (double) std::max<size_t>(n, 1));
        return s;
    }
};

static int topk_overlap(const float * a, const float * b, size_t n, int k) {
    std::vector<size_t> ia(n), ib(n);
    for (size_t i = 0; i < n; ++i) ia[i] = ib[i] = i;
    std::partial_sort(ia.begin(), ia.begin() + k, ia.end(), [&](size_t x, size_t y) { return a[x] > a[y]; });
    std::partial_sort(ib.begin(), ib.begin() + k, ib.end(), [&](size_t x, size_t y) { return b[x] > b[y]; });
    int hit = 0;
    for (int i = 0; i < k; ++i) for (int j = 0; j < k; ++j) if (ia[i] == ib[j]) { ++hit; break; }
    return hit;
}

static void print_err(const char * tag, const ErrStats & s, int topk_hit = -1, int topk = 0) {
    std::cout << "  [" << tag << "] n=" << s.n << " max_abs=" << s.max_abs << " rmse=" << s.rmse;
    if (s.has_nan) std::cout << " (NaN present!)";
    if (topk_hit >= 0) std::cout << " top" << topk << "_hit=" << topk_hit << "/" << topk;
    std::cout << "\n";
}

// ============================ harness ============================

namespace {

constexpr int kMaxNodes = 16384;

// 容差：音频塔 F16 权重 + 18 层累积，参考 fp32。阈值按首版实测建立（见 --report）。
constexpr float kAudioRmseTol   = 0.20f; // conv_out / 逐层 hidden / features 的 rmse 上限
constexpr float kEmbRmseTol     = 0.05f; // 融合 Embedding rmse 上限（与 v0.1 逐层 hidden 同量级）
constexpr float kLogitsMaxTol   = 0.5f;  // 末位 logits max abs 上限
constexpr int   kTopKMin        = 9;     // top10 最小命中数

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

    // 取单个 token 的文本 Embedding（tok_embd F16 -> F32），写入 dst[n_embd]。
    void token_embd(int32_t id, float * dst) const {
        const int32_t n_embd = model.hparams.n_embd;
        std::vector<ggml_fp16_t> row((size_t) n_embd);
        ggml_backend_tensor_get(model.tok_embd, row.data(),
                                (size_t) id * n_embd * sizeof(ggml_fp16_t),
                                (size_t) n_embd * sizeof(ggml_fp16_t));
        for (int32_t i = 0; i < n_embd; ++i) dst[i] = ggml_fp16_to_fp32(row[i]);
    }

    struct ggml_cgraph * eval(ggml_context * ctx, const int32_t * tokens,
                              const float * embd_in, int32_t n_tokens, int32_t n_past,
                              std::vector<float> & logits_out) {
        qwen3_graph_params gp;
        gp.n_tokens = n_tokens; gp.n_past = n_past; gp.max_nodes = kMaxNodes;
        gp.logits_last_only = true; gp.want_layer_outputs = false;

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
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_TOKENS),
                                    tokens, 0, (size_t) n_tokens * sizeof(int32_t));
        }
        std::vector<int32_t> pos(n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) pos[i] = n_past + i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, QWEN3_TENSOR_NAME_POS),
                                pos.data(), 0, (size_t) n_tokens * sizeof(int32_t));
        const int32_t n_kv = n_past + n_tokens;
        std::vector<float> mask((size_t) n_tokens * n_kv, 0.0f);
        for (int32_t q = 0; q < n_tokens; ++q)
            for (int32_t k = n_past + q + 1; k < n_kv; ++k)
                mask[(size_t) q * n_kv + k] = -INFINITY;
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
        for (int32_t e : model.asr_hparams.eos_token_ids) if (id == e) return true;
        return false;
    }
};

// 构造 ASR prompt（官方 chat_template，单音频；不加 Thinking 前缀）。
// token 名与官方词表一致：<|audio_start|>（151669）/ <|audio_end|>（151670）/<|audio_pad|>（151676）。
std::string build_prompt(const std::string & mode, int32_t audio_pad_count,
                         const std::string & language, const std::string & context) {
    std::string pad;
    for (int32_t i = 0; i < audio_pad_count; ++i) pad += "<|audio_pad|>";
    std::string prompt = "<|im_start|>system\n" + context + "<|im_end|>\n"
                       + "<|im_start|>user\n<|audio_start|>" + pad + "<|audio_end|><|im_end|>\n"
                       + "<|im_start|>assistant\n";
    if (mode == "lang") prompt += "language " + language + "<asr_text>";
    return prompt;
}

} // namespace

// ============================ main ============================

int main(int argc, char ** argv) {
    std::string gguf_path, ref_dir, mode = "auto", language = "Chinese", context;
    int max_gen = 32;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--ref-dir" && i + 1 < argc) ref_dir = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--language" && i + 1 < argc) language = argv[++i];
        else if (a == "--context" && i + 1 < argc) context = argv[++i];
        else if (a == "--max-gen" && i + 1 < argc) max_gen = std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " --gguf <path> --ref-dir <dir> --mode auto|lang"
                      << " [--language Chinese] [--context str] [--max-gen N]\n";
            return 0;
        }
    }
    if (gguf_path.empty() || ref_dir.empty()) { std::cerr << "missing --gguf / --ref-dir\n"; return 1; }

    // ---- 参考数据 ----
    const NpyArray ref_mel   = load_npy(ref_dir + "/mel.npy");            // [128, T]
    const NpyArray ref_ids   = load_npy(ref_dir + "/ids.npy");            // [S]
    const NpyArray ref_apos  = load_npy(ref_dir + "/audio_positions.npy");// [A]
    const NpyArray ref_cout  = load_npy(ref_dir + "/audio_conv_out.npy"); // [A, 896]
    const NpyArray ref_ahid  = load_npy(ref_dir + "/audio_hidden.npy");   // [19, A, 896]
    const NpyArray ref_afeat = load_npy(ref_dir + "/audio_features.npy"); // [A, 1024]
    const NpyArray ref_embd  = load_npy(ref_dir + "/embd.npy");           // [S, 1024]
    const NpyArray ref_dlogits = load_npy(ref_dir + "/decode_logits.npy");// [n_gen, vocab]
    const NpyArray ref_dids  = load_npy(ref_dir + "/decode_ids.npy");     // [n_gen]

    const int32_t n_mel = (int32_t) ref_mel.shape[0];
    const int32_t T     = (int32_t) ref_mel.shape[1];
    const int32_t A_ref = (int32_t) ref_apos.shape[0];
    const int32_t S_ref = (int32_t) ref_ids.shape[0];
    std::cout << "== asr/v0.2 alignment ==\n  mode=" << mode
              << " T(mel)=" << T << " A_ref=" << A_ref << " S_ref=" << S_ref
              << " n_gen_ref=" << ref_dids.shape[0] << "\n";

    // 预分配 n_ctx（prefill + 生成上限）
    Harness hs;
    const int32_t n_ctx = S_ref + max_gen + 16;
    if (!hs.load(gguf_path, n_ctx)) { std::cerr << "failed to load model from " << gguf_path << "\n"; return 1; }
    const int32_t n_vocab = hs.model.hparams.n_vocab;
    const int32_t n_embd  = hs.model.hparams.n_embd;
    if (n_mel != hs.model.asr_hparams.num_mel_bins) {
        std::cerr << "mel bins mismatch: ref " << n_mel << " vs gguf " << hs.model.asr_hparams.num_mel_bins << "\n";
        return 1;
    }
    int failures = 0;

    // ================= 1. 音频 Encoder（ggml 音频塔） =================
    std::cout << "-- audio encoder --\n";
    qwen3_asr_audio_result audio;
    if (!qwen3_asr_encode_audio(hs.model, ref_mel.as_f32(), T, /*want_intermediates=*/true, audio)) {
        std::cerr << "audio encoder failed\n"; return 1;
    }
    std::cout << "  A(cpp)=" << audio.A << " (ref " << A_ref << ")\n";
    if (audio.A != A_ref) { std::cout << "  [A mismatch] FAIL\n"; ++failures; }

    const int32_t A = audio.A;
    const int32_t d_model = hs.model.asr_hparams.d_model;
    const int32_t n_alayer = hs.model.asr_hparams.n_audio_layer;

    // CNN 输出（conv_out）
    {
        const ErrStats s = ErrStats::compute(audio.conv_out.data(), ref_cout.as_f32(), (size_t) A * d_model);
        print_err("conv_out", s);
        if (s.rmse > kAudioRmseTol) ++failures;
    }
    // 音频 Transformer 逐层（hidden[0]=conv_out，hidden[i]=第 i 层输出）
    {
        double worst = 0.0; int worst_il = -1;
        for (int il = 1; il <= n_alayer; ++il) {
            const float * cpp = audio.hidden.data() + (size_t) il * A * d_model;
            const float * ref = ref_ahid.as_f32() + (size_t) il * A * d_model;
            const ErrStats s = ErrStats::compute(cpp, ref, (size_t) A * d_model);
            if (s.rmse > worst) { worst = s.rmse; worst_il = il; }
            std::cout << "  [audio_layer " << (il - 1) << "] max_abs=" << s.max_abs << " rmse=" << s.rmse << "\n";
        }
        std::cout << "  worst audio layer=" << worst_il << " rmse=" << worst << "\n";
        if (worst > kAudioRmseTol) ++failures;
    }
    // 投影输出 features [A,1024]
    {
        const ErrStats s = ErrStats::compute(audio.features.data(), ref_afeat.as_f32(), (size_t) A * n_embd);
        print_err("audio_features", s);
        if (s.rmse > kAudioRmseTol) ++failures;
    }

    // ================= 2. ASR Prompt + Tokenizer =================
    std::cout << "-- prompt / tokenize --\n";
    const std::string prompt = build_prompt(mode, A, language, context);
    // std::cout << "  prompt: " << prompt << "\n";
    const std::vector<int32_t> ids = hs.tok.encode(prompt);
    const int32_t S = (int32_t) ids.size();
    std::cout << "  S(cpp)=" << S << " (ref " << S_ref << ")\n";
    {
        bool ids_ok = (S == S_ref);
        if (ids_ok) for (int i = 0; i < S; ++i) ids_ok = ids_ok && (ids[i] == ref_ids.as_i32()[i]);
        std::cout << "  [ids] " << (ids_ok ? "match" : "MISMATCH") << "\n";
        if (!ids_ok) ++failures;
    }
    // audio_pad 位置
    const int32_t audio_pad_id = hs.model.asr_hparams.audio_token_id;
    std::vector<int32_t> apos;
    for (int32_t i = 0; i < S; ++i) if (ids[i] == audio_pad_id) apos.push_back(i);
    std::cout << "  audio_pad count=" << apos.size() << " (A=" << A << ")\n";
    if ((int32_t) apos.size() != A) { std::cout << "  [audio_pad count] FAIL\n"; ++failures; }

    // ================= 3. 融合（替换 audio_pad 位置的文本 Embedding） =================
    std::cout << "-- fusion --\n";
    std::vector<float> fused((size_t) S * n_embd, 0.0f);
    {
        size_t ai = 0;
        for (int32_t s = 0; s < S; ++s) {
            float * row = fused.data() + (size_t) s * n_embd;
            if (ids[s] == audio_pad_id) {
                if (ai < (size_t) A) { // audio_pad 数应等于 A；防御越界
                    std::memcpy(row, audio.features.data() + (size_t) ai * n_embd, (size_t) n_embd * sizeof(float));
                }
                ++ai;
            } else {
                hs.token_embd(ids[s], row);
            }
        }
        // ids 长度不一致时参考 embd 只有 S_ref 行，按最小长度比较，避免越界读
        const int32_t S_cmp = std::min(S, S_ref);
        const ErrStats e = ErrStats::compute(fused.data(), ref_embd.as_f32(), (size_t) S_cmp * n_embd);
        print_err("fused_embd", e);
        if (e.rmse > kEmbRmseTol || S != S_ref) ++failures;
    }

    // ================= 4. prefill + 末位 logits =================
    std::cout << "-- prefill --\n";
    std::vector<float> logits;
    {
        struct ggml_init_params gp = {
            /*.mem_size =*/ ggml_tensor_overhead() * kMaxNodes * 4 + ggml_graph_overhead_custom(kMaxNodes, false),
            /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true,
        };
        struct ggml_context * ctx = ggml_init(gp);
        hs.eval(ctx, nullptr, fused.data(), S, 0, logits);
        ggml_free(ctx);
        const float * ref = ref_dlogits.as_f32(); // 第 0 行 = prefill 末位 logits
        const ErrStats ls = ErrStats::compute(logits.data(), ref, (size_t) n_vocab);
        const int hit = topk_overlap(logits.data(), ref, (size_t) n_vocab, 10);
        print_err("prefill_logits", ls, hit, 10);
        if (ls.max_abs > kLogitsMaxTol || hit < kTopKMin) ++failures;
    }

    // ================= 5. teacher forcing 多步 decode =================
    {
        const int n_gen = (int) ref_dids.shape[0];
        std::cout << "-- teacher forcing (" << n_gen - 1 << " steps) --\n";
        double worst_max = 0.0; int worst_top = 10;
        const int32_t * dids = ref_dids.as_i32();
        for (int i = 1; i < n_gen; ++i) {
            struct ggml_init_params gp = {
                /*.mem_size =*/ ggml_tensor_overhead() * kMaxNodes * 4 + ggml_graph_overhead_custom(kMaxNodes, false),
                /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true,
            };
            struct ggml_context * ctx = ggml_init(gp);
            hs.eval(ctx, dids + (i - 1), nullptr, 1, S + i - 1, logits);
            ggml_free(ctx);
            const float * ref = ref_dlogits.as_f32() + (size_t) i * n_vocab;
            const ErrStats s = ErrStats::compute(logits.data(), ref, (size_t) n_vocab);
            worst_max = std::max(worst_max, s.max_abs);
            worst_top = std::min(worst_top, topk_overlap(logits.data(), ref, (size_t) n_vocab, 10));
        }
        std::cout << "  [decode_logits] worst_max_abs=" << worst_max << " worst_top10_hit=" << worst_top << "/10\n";
        if (worst_max > kLogitsMaxTol || worst_top < kTopKMin) ++failures;
    }

    // ================= 6. 自由贪心生成 + 文本解析 =================
    {
        std::cout << "-- free greedy generation --\n";
        hs.kv.n_past = 0;
        const int n_gen = (int) ref_dids.shape[0];
        const int32_t * dids = ref_dids.as_i32();
        std::vector<int32_t> gen;
        int32_t n_past = 0;
        for (int step = 0; step < max_gen; ++step) {
            struct ggml_init_params gp = {
                /*.mem_size =*/ ggml_tensor_overhead() * kMaxNodes * 4 + ggml_graph_overhead_custom(kMaxNodes, false),
                /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true,
            };
            struct ggml_context * ctx = ggml_init(gp);
            if (step == 0) hs.eval(ctx, nullptr, fused.data(), S, 0, logits);
            else           hs.eval(ctx, &gen.back(), nullptr, 1, n_past, logits);
            n_past = hs.kv.n_past;
            ggml_free(ctx);
            int32_t best = 0; float bv = logits[0];
            for (int32_t v = 1; v < n_vocab; ++v) if (logits[v] > bv) { bv = logits[v]; best = v; }
            gen.push_back(best);
            if (hs.is_eos(best)) break;
        }
        const int match_n = std::min(gen.size(), (size_t) n_gen);
        int token_match = 0;
        for (int i = 0; i < match_n; ++i) token_match += (gen[i] == dids[i]) ? 1 : 0;
        std::cout << "  [free_gen] n=" << gen.size() << " (ref " << n_gen << ") prefix_match="
                  << token_match << "/" << match_n << "\n";

        // 原始输出 + ASR 协议解析（language + text）
        const std::string raw = hs.tok.decode(gen, /*skip_special=*/false);
        const std::string body = hs.tok.decode(gen, /*skip_special=*/true);
        std::cout << "  [raw ] " << raw << "\n";
        std::cout << "  [text] " << body << "\n";
        if (gen.size() != (size_t) n_gen || token_match != match_n) ++failures;
    }

    std::cout << "== result: " << (failures == 0 ? "PASS" : "FAIL") << " (failures=" << failures << ") ==\n";
    return failures == 0 ? 0 : 1;
}
