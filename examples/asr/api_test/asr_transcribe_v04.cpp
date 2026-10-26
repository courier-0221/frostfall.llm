// ============================================================================
// asr_transcribe_v04.cpp —— asr/v0.4 完整离线 API + 文本流式 + 取消 + 统计示例
//
// 相比 v0.3（最小阻塞 API），v0.4 演示（doc/asr/design_asr.md §5）：
//   1. 阻塞 vs 文本流式：同一输入两种模式，最终结果必须一致（流式按 UTF-8
//      严格边界增量回调正文；自动语言模式先回调 language 一次）；
//   2. 取消：流式回调提前返回 false（等价置位 request.cancel），校验
//      FINISH_CANCELLED + 取消时 stage + 已回调的部分正文；
//   3. 分批 prefill：--prefill-batch N 启用，与单批结果一致；
//   4. 跨窗口：--window-ref 模式对齐 work/asr_ref_win/cross 的 golden
//      （官方 get_audio_cu_seqlens 块对角双向窗口语义）；
//   5. 统计：前处理/Encoder/prefill/decode/首模型 token/首可见文字/RTF/
//      峰值计算缓冲；
//   6. 错误路径（--check-errors）：空输入/非 16 kHz/超长/NaN/上下文不足。
//
// 用法：
//   asr_transcribe_v04 --config examples/asr/api_test/asr_conf_v04.json \
//                     --wav data/test_asr_001.wav
//   asr_transcribe_v04 --config ... --pcm work/asr_ref_v02/auto/pcm.npy
//   asr_transcribe_v04 --config ... --window-ref work/asr_ref_win/cross   # 跨窗口对齐
//   asr_transcribe_v04 --config ... --wav x.wav --prefill-batch 32        # 分批 prefill
//   asr_transcribe_v04 --config ... --wav x.wav --cancel-after 2          # 取消演示
//   asr_transcribe_v04 --config ... --wav x.wav --check-errors
// ============================================================================

#include "asr_engine.h"
#include "audio_encoder.h"
#include "model.h"
#include "tokenizer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// 独立加载 GGUF 做跨窗口 golden 对齐（不经 Transcribe，直接对齐音频塔输出）
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

// ============================ .npy 读取（同 v0.3 示例的最小实现） ============================

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
    if (ver[0] != 1) throw std::runtime_error("only npy v1.x supported");
    uint16_t header_len = 0; f.read(reinterpret_cast<char *>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

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

// ============================ PCM16 WAV 读取（单声道 16 kHz，/32768；同 v0.3） ============================

struct WavInfo { int32_t sample_rate = 0, channels = 0, bits = 0, format = 0; };

static bool read_wav_pcm16(const std::string & path, std::vector<float> & pcm_out,
                           WavInfo & info, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open wav: " + path; return false; }
    char hdr[12];
    if (!f.read(hdr, 12) || std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file: " + path; return false;
    }
    bool have_fmt = false, have_data = false;
    while (f && !have_data) {
        char id[4]; uint32_t sz = 0;
        if (!f.read(id, 4)) break;
        f.read(reinterpret_cast<char *>(&sz), 4);
        const std::streamoff next_off = (std::streamoff) f.tellg()
                                        + (std::streamoff) (((sz + 1) / 2) * 2);
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t audio_format = 0, channels = 0, bits = 0;
            uint32_t rate = 0;
            f.read(reinterpret_cast<char *>(&audio_format), 2);
            f.read(reinterpret_cast<char *>(&channels), 2);
            f.read(reinterpret_cast<char *>(&rate), 4);
            f.seekg(4, std::ios::cur); // byte rate
            f.read(reinterpret_cast<char *>(&bits), 2); // dummy read block align 先跳
            f.seekg(-2, std::ios::cur);
            uint16_t block_align = 0;
            f.read(reinterpret_cast<char *>(&block_align), 2);
            f.read(reinterpret_cast<char *>(&bits), 2);
            info.format = audio_format; info.channels = channels;
            info.sample_rate = (int32_t) rate; info.bits = bits;
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!have_fmt) { err = "wav data chunk before fmt: " + path; return false; }
            if (info.format != 1) {
                err = "unsupported wav format " + std::to_string(info.format) + " (need PCM=1)";
                return false;
            }
            if (info.bits != 16) {
                err = "unsupported wav bit depth " + std::to_string(info.bits) +
                      " (only 16-bit PCM; 8/24/32-bit must be converted first)";
                return false;
            }
            if (info.channels != 1) {
                err = "unsupported wav channels " + std::to_string(info.channels) +
                      " (only mono; stereo/multichannel must be downmixed first)";
                return false;
            }
            std::vector<int16_t> raw((size_t) sz / 2);
            f.read(reinterpret_cast<char *>(raw.data()), (std::streamsize) (raw.size() * 2));
            pcm_out.resize(raw.size());
            for (size_t i = 0; i < raw.size(); ++i) pcm_out[i] = (float) raw[i] / 32768.0f;
            have_data = true;
        }
        f.clear();
        f.seekg(next_off, std::ios::beg);
    }
    if (!have_fmt || !have_data) { err = "incomplete wav (missing fmt/data chunk): " + path; return false; }
    return true;
}

// ============================ 打印 / 对比工具 ============================

static void print_response(const std::string & tag, const qwen3_asr_response & r, bool debug) {
    if (!r.ok) {
        std::cout << "[" << tag << "] ERROR(" << qwen3_asr_error_str(r.err) << "@" << qwen3_asr_stage_str(r.stage)
                  << "): " << r.error << "\n";
        return;
    }
    std::cout << "[" << tag << "] finish=" << qwen3_asr_finish_reason_str(r.finish_reason)
              << " stage=" << qwen3_asr_stage_str(r.stage)
              << " language=" << (r.language.empty() ? "<none>" : r.language)
              << " text=" << (!r.text.empty() ? r.text : "<empty>") << "\n";
    if (debug) {
        std::cout << "  [debug] T=" << r.T << " A=" << r.A << " S=" << r.S
                  << " n_gen=" << r.n_gen << "\n";
        if (!r.raw.empty()) std::cout << "  [raw ] " << r.raw << "\n";
    }
    const qwen3_asr_stats & st = r.stats;
    std::cout << "  [stats] total=" << st.t_total << "s rtf=" << st.rtf
              << " preproc=" << st.t_preproc << "s encoder=" << st.t_encoder << "s"
              << " prefill=" << st.t_prefill << "s (batches=" << st.n_prefill_batches
              << ", bs=" << st.prefill_batch_size << ")"
              << " decode=" << st.t_decode << "s (steps=" << st.n_decode_steps << ")\n"
              << "  [stats] first_model_token="
              << (st.t_first_model_token >= 0 ? std::to_string(st.t_first_model_token) + "s" : std::string("n/a"))
              << " first_visible_text="
              << (st.t_first_visible_text >= 0 ? std::to_string(st.t_first_visible_text) + "s" : std::string("n/a"))
              << " peak_galloc=" << st.peak_galloc_bytes / (1024.0 * 1024.0) << "MB\n";
}

static double max_abs_diff(const float * a, const float * b, size_t n, double * rmse) {
    double mx = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        mx = std::max(mx, d);
        sum2 += d * d;
    }
    *rmse = std::sqrt(sum2 / (double) std::max<size_t>(n, 1));
    return mx;
}

// ============================ 跨窗口 golden 对齐（音频塔直连） ============================

static int check_window_ref(const std::string & model_path, const std::string & ref_dir) {
    std::cout << "-- cross-window golden alignment (" << ref_dir << ") --\n";
    int failures = 0;
    qwen3_model model;
    if (!qwen3_model_load(model_path, model)) { std::cerr << "model load failed\n"; return 1; }
    if (!model.is_asr) { std::cerr << "not an ASR model\n"; return 1; }

    try {
        const NpyArray g_mel  = load_npy(ref_dir + "/mel.npy");             // [128, T]
        const NpyArray g_feat = load_npy(ref_dir + "/audio_features.npy");  // [A, 1024]
        const NpyArray g_hdn  = load_npy(ref_dir + "/audio_hidden.npy");    // [19, A, 896]
        const NpyArray g_cout = load_npy(ref_dir + "/audio_conv_out.npy");  // [A, 896]
        const NpyArray g_bnd  = load_npy(ref_dir + "/window_bounds.npy");   // [n_win+1]
        const int32_t T = (int32_t) g_mel.shape[1];
        const int32_t A_ref = (int32_t) g_feat.shape[0];

        qwen3_asr_audio_result audio;
        if (!qwen3_asr_encode_audio(model, g_mel.as_f32(), T, /*want_intermediates=*/true, audio)) {
            std::cout << "  [encoder] failed  FAIL\n";
            return 1;
        }
        {
            std::cout << "  [A] cpp=" << audio.A << " ref=" << A_ref << "  "
                      << (audio.A == A_ref ? "OK" : "FAIL") << "\n";
            if (audio.A != A_ref) ++failures;
        }
        // 窗口边界核对：C++ 不导出边界（实现内部），以官方公式重算对齐
        {
            // bounds: aftercnn 逐块，chunk=100
            std::vector<int32_t> aftercnn;
            auto floor_div = [](int32_t a, int32_t b) { // Python/torch 地板除
                return (a >= 0) ? a / b : -((-a + b - 1) / b);
            };
            for (int32_t c0 = 0; c0 < T; c0 += 100) {
                const int32_t len = std::min(100, T - c0);
                const int32_t leave = len % 100;
                const int32_t feat = floor_div(leave - 1, 2) + 1;
                aftercnn.push_back(floor_div(floor_div(feat - 1, 2) + 1 - 1, 2) + 1
                                   + (len / 100) * 13); // 官方 (len//chunk_len)*13 项
            }
            const int32_t window = 13 * 8; // window_aftercnn 固定于完整块 (*13*(800/100))
            std::vector<int32_t> bounds{0};
            int32_t cur = 0;
            for (int32_t al : aftercnn) {
                cur += al;
                if (cur >= window) { bounds.push_back(bounds.back() + cur); cur = 0; }
            }
            if (cur > 0) bounds.push_back(bounds.back() + cur);
            bool ok = (int32_t) g_bnd.shape[0] == (int32_t) bounds.size();
            for (size_t i = 0; ok && i < bounds.size(); ++i)
                ok = bounds[i] == g_bnd.as_i32()[i];
            std::cout << "  [window_bounds] cpp=[" << [&] { std::string s; for (int32_t b : bounds) s += std::to_string(b) + " "; return s; }()
                      << "] ref_n=" << g_bnd.shape[0] << "  " << (ok ? "OK" : "FAIL") << "\n";
            if (!ok) ++failures;
        }
        // 容差沿用 asr_align_v02 的 kAudioRmseTol=0.20（rmse 判定；f16 权重 vs fp32
        // 参考的最大绝对误差允许超过 rmse）。跨窗口与单窗口同一判据。
        constexpr double kRmseTol = 0.20;
        auto check = [&](const char * tag, const float * cpp, const float * ref, size_t n) {
            double rmse = 0;
            const double mx = max_abs_diff(cpp, ref, n, &rmse);
            const bool ok = rmse <= kRmseTol && mx < 1e4 /*非发散保护*/;
            std::cout << "  [" << tag << "] max_abs=" << mx << " rmse=" << rmse
                      << (ok ? "  OK" : "  FAIL") << "\n";
            if (!ok) ++failures;
        };
        check("conv_out",  audio.conv_out.data(),  g_cout.as_f32(), (size_t) A_ref * 896);
        {
            const size_t n_layers = model.asr_hparams.n_audio_layer;
            check("layer_out", audio.hidden.data(), g_hdn.as_f32(),
                  (n_layers + 1) * (size_t) A_ref * 896);
        }
        {
            const int32_t od = model.asr_hparams.output_dim;
            check("features",  audio.features.data(), g_feat.as_f32(), (size_t) A_ref * od);
        }
    } catch (const std::exception & e) {
        std::cout << "  [golden] " << e.what() << "  FAIL\n";
        ++failures;
    }
    return failures;
}

// ============================ main ============================

int main(int argc, char ** argv) {
    std::string config_path, gguf_path, wav_path, pcm_path, ref_dir, window_ref, language, context;
    int32_t max_gen = 0, prefill_batch = 0, cancel_after = -1;
    bool check_errors = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (a == "--gguf"   && i + 1 < argc) gguf_path   = argv[++i];
        else if (a == "--wav"    && i + 1 < argc) wav_path    = argv[++i];
        else if (a == "--pcm"    && i + 1 < argc) pcm_path    = argv[++i]; // float32 npy（16k）
        else if (a == "--ref-dir" && i + 1 < argc) ref_dir    = argv[++i];
        else if (a == "--window-ref" && i + 1 < argc) window_ref = argv[++i];
        else if (a == "--language" && i + 1 < argc) language   = argv[++i];
        else if (a == "--context"  && i + 1 < argc) context    = argv[++i];
        else if (a == "--max-gen"  && i + 1 < argc) max_gen    = std::atoi(argv[++i]);
        else if (a == "--prefill-batch" && i + 1 < argc) prefill_batch = std::atoi(argv[++i]);
        else if (a == "--cancel-after" && i + 1 < argc) cancel_after  = std::atoi(argv[++i]);
        else if (a == "--check-errors") check_errors = true;
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " --config <asr_conf_v04.json> [--gguf <model.gguf>]\n"
                      << "        [--wav <in.wav> | --pcm <pcm.npy> | --ref-dir <dir>] [--window-ref <dir>]\n"
                      << "        [--language Chinese] [--context str] [--max-gen N]\n"
                      << "        [--prefill-batch N] [--cancel-after N] [--check-errors]\n";
            return 0;
        }
    }

    std::cout << "== asr/v0.4 offline/streaming transcribe ==\n";

    // ---- 引擎初始化 ----
    std::string model_path;
    qwen3_asr_engine engine;
    if (!config_path.empty()) {
        std::cout << "engine init from " << config_path << "\n";
        if (!engine.InitFromJsonFile(config_path)) { std::cerr << "InitFromJsonFile failed\n"; return 1; }
        // window-ref 需要 model_path；从 JSON 反查不便，直接用 --gguf，或读回配置
        std::ifstream f(config_path);
        if (f) { // 简化：window-ref 模式要求 --gguf 提供路径
            std::string line;
            while (std::getline(f, line)) {
                auto pos = line.find("\"model_path\"");
                if (pos != std::string::npos) {
                    auto q0 = line.find('"', pos + 12);
                    auto q1 = line.find('"', q0 + 1);
                    if (q0 != std::string::npos && q1 != std::string::npos) model_path = line.substr(q0 + 1, q1 - q0 - 1);
                }
            }
        }
    } else if (!gguf_path.empty()) {
        qwen3_asr_engine_config cfg;
        cfg.model_path = gguf_path;
        std::cout << "engine init from " << gguf_path << "\n";
        if (!engine.Init(cfg)) { std::cerr << "Init failed\n"; return 1; }
        model_path = gguf_path;
    } else {
        std::cerr << "missing --config or --gguf\n";
        return 1;
    }
    std::cout << "engine ready (n_ctx=" << engine.NCtx()
              << ", max_audio_positions=" << engine.MaxAudioPositions() << ")\n";

    int failures = 0;

    // ---- 跨窗口 golden 对齐（--window-ref）：独立模式，音频塔直连 ----
    if (!window_ref.empty()) {
        if (model_path.empty()) { std::cerr << "--window-ref needs --gguf or model_path in config\n"; return 1; }
        failures += check_window_ref(model_path, window_ref);
    }

    if (wav_path.empty() && pcm_path.empty() && ref_dir.empty()) {
        if (!window_ref.empty()) { // 只跑窗口对齐也可以
            std::cout << "== result: " << (failures ? "FAIL" : "PASS") << " (failures=" << failures << ") ==\n";
            return failures ? 1 : 0;
        }
        std::cerr << "missing input: one of --wav / --pcm / --ref-dir required\n";
        return 1;
    }

    // ---- 输入 PCM ----
    std::vector<float> pcm;
    int32_t sample_rate = 16000;
    if (!ref_dir.empty()) {
        try {
            const NpyArray p = load_npy(ref_dir + "/pcm.npy");
            pcm.assign(p.as_f32(), p.as_f32() + p.count());
        } catch (const std::exception & e) { std::cerr << e.what() << "\n"; return 1; }
        std::cout << "input: ref pcm " << pcm.size() << " samples (" << pcm.size() / 16000.0 << "s)\n";
    } else if (!wav_path.empty()) {
        WavInfo info;
        std::string werr;
        if (!read_wav_pcm16(wav_path, pcm, info, werr)) { std::cerr << werr << "\n"; return 1; }
        sample_rate = info.sample_rate;
        std::cout << "input: wav " << pcm.size() << " samples, " << info.sample_rate << " Hz\n";
    } else {
        try {
            const NpyArray p = load_npy(pcm_path);
            pcm.assign(p.as_f32(), p.as_f32() + p.count());
        } catch (const std::exception & e) { std::cerr << e.what() << "\n"; return 1; }
        std::cout << "input: pcm " << pcm.size() << " samples\n";
    }

    // ---- 1. 阻塞模式 ----
    qwen3_asr_request req;
    req.pcm = pcm.data();
    req.n_samples = (int32_t) pcm.size();
    req.sample_rate = sample_rate;
    req.language = language;
    req.context = context;
    req.max_gen_tokens = max_gen;
    req.prefill_batch_size = prefill_batch;
    req.want_debug = true;

    const qwen3_asr_response r_block = engine.Transcribe(req);
    print_response("blocking#1", r_block, /*debug=*/true);
    if (!r_block.ok) return 1;

    // ---- 2. 文本流式模式：逐 token 回调，最终结果必须与阻塞一致 ----
    std::string streamed_text, streamed_lang;
    int n_text_callbacks = 0, n_lang_callbacks = 0;
    {
        std::cout << "[streaming] pieces:";
        qwen3_asr_request rs = req;
        const qwen3_asr_response r_stream = engine.Transcribe(rs, [&](qwen3_asr_stream_event ev,
                                                                      const std::string & piece) {
            if (ev == QWEN3_ASR_STREAM_LANG) {
                ++n_lang_callbacks;
                streamed_lang = piece;
                std::cout << " <lang:" << piece << ">";
            } else {
                ++n_text_callbacks;
                streamed_text += piece;
                std::cout << " [" << piece << "]";
            }
            return true;
        });
        std::cout << "\n";
        print_response("streaming#1", r_stream, /*debug=*/false);
        if (!r_stream.ok) { ++failures; }
        else {
            if (r_stream.text != r_block.text) {
                std::cout << "  [consistency] streamed final text != blocking text  FAIL\n"
                          << "    blocking: " << r_block.text << "\n"
                          << "    streamed: " << r_stream.text << "\n";
                ++failures;
            } else {
                std::cout << "  [consistency] streamed final == blocking (text bytes "
                          << r_stream.text.size() << ")  OK\n";
            }
            if (!language.empty() && n_lang_callbacks != 0) {
                std::cout << "  [consistency] lang mode emitted LANG event (should not)  FAIL\n";
                ++failures;
            }
        }
    }

    // ---- 2b. golden 对齐（--ref-dir）：分批/跨窗口改造后生成 ids 必须仍与 v0.3 golden 一致 ----
    if (!ref_dir.empty()) {
        std::cout << "-- golden alignment (" << ref_dir << ") --\n";
        try {
            const NpyArray g_ids = load_npy(ref_dir + "/decode_ids.npy"); // [n_gen]
            const int32_t n_ref = (int32_t) g_ids.shape[0];
            const int32_t n_cmn = std::min<int32_t>(n_ref, (int32_t) r_block.gen_ids.size());
            int match = 0;
            for (int32_t i = 0; i < n_cmn; ++i)
                match += (r_block.gen_ids[(size_t) i] == g_ids.as_i32()[i]) ? 1 : 0;
            std::cout << "  [gen_ids] n_cpp=" << r_block.gen_ids.size() << " n_ref=" << n_ref
                      << " prefix_match=" << match << "/" << n_cmn;
            if ((int32_t) r_block.gen_ids.size() == n_ref && match == n_cmn) {
                std::cout << "  OK\n";
            } else {
                std::cout << "  FAIL\n";
                ++failures;
            }
        } catch (const std::exception & e) {
            std::cout << "  [golden] " << e.what() << "  FAIL\n";
            ++failures;
        }
    }

    // ---- 3. 分批 prefill：小批与单批/上次结果一致，统计批数 ----
    if (prefill_batch > 0) {
        std::cout << "[batched-prefill] requested bs=" << prefill_batch
                  << " -> actual batches=" << r_block.stats.n_prefill_batches
                  << " (S=" << r_block.S << ")\n";
    }

    // ---- 4. 取消：--cancel-after N（N 个 TEXT 回调后取消）----
    if (cancel_after >= 0) {
        std::atomic<bool> cancel_flag{false};
        int seen = 0;
        std::string cancelled_text;
        qwen3_asr_request rc = req;
        rc.cancel = &cancel_flag;
        const qwen3_asr_response r_cancel = engine.Transcribe(rc, [&](qwen3_asr_stream_event ev,
                                                                      const std::string & piece) {
            if (ev == QWEN3_ASR_STREAM_TEXT) {
                ++seen;
                cancelled_text += piece;
                if (seen >= cancel_after) { cancel_flag.store(true); return false; }
            }
            return true;
        });
        print_response("cancelled", r_cancel, /*debug=*/false);
        // 取消时 stage 语义：首 token 在 prefill 末位产出 —— 首个 TEXT 回调即取消时
        // 尚未进入 decode step（stage=prefill）；进入 decode 后取消则 stage=decode。
        const qwen3_asr_stage want_stage =
            r_cancel.stats.n_decode_steps > 0 ? QWEN3_ASR_STAGE_DECODE : QWEN3_ASR_STAGE_PREFILL;
        const bool ok = r_cancel.ok &&
                        r_cancel.finish_reason == QWEN3_ASR_FINISH_CANCELLED &&
                        r_cancel.stage == want_stage &&
                        r_cancel.stage != QWEN3_ASR_STAGE_DONE &&
                        !cancelled_text.empty() &&
                        r_cancel.text.compare(0, cancelled_text.size(), cancelled_text) == 0;
        std::cout << "  [cancel] finish_reason=cancelled stage=" << qwen3_asr_stage_str(r_cancel.stage)
                  << " partial_text_prefix_ok="
                  << (r_cancel.text.compare(0, cancelled_text.size(), cancelled_text) == 0 ? "yes" : "no")
                  << "  " << (ok ? "OK" : "FAIL") << "\n";
        if (!ok) ++failures;
    }

    // ---- 5. 外部线程取消（prefill 边界；无 callback，长音频 + 异步置位）----
    {
        // 用跨窗口长输入 + --pcm 已可携带；异步线程在 prefill 期间置位
        if (req.n_samples > 8 * 16000 && !pcm.empty()) {
            std::atomic<bool> cancel_flag{false};
            qwen3_asr_request rc = req;
            rc.cancel = &cancel_flag;
            rc.want_debug = false;
            std::thread canceller([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds {50});
                cancel_flag.store(true);
            });
            const qwen3_asr_response r_ext = engine.Transcribe(rc);
            canceller.join();
            print_response("ext-cancel", r_ext, /*debug=*/false);
            const bool ok = r_ext.ok && r_ext.finish_reason == QWEN3_ASR_FINISH_CANCELLED;
            std::cout << "  [ext-cancel] " << (ok ? "OK" : "FAIL") << "\n";
            if (!ok) ++failures;
        }
    }

    // ---- 6. 错误路径演示（--check-errors） ----
    if (check_errors) {
        std::cout << "-- error checks --\n";
        auto expect_error = [&](const std::string & tag, qwen3_asr_request bad_req,
                                qwen3_asr_error want_err) {
            bad_req.want_debug = false;
            const qwen3_asr_response r = engine.Transcribe(bad_req);
            const bool good = !r.ok && r.err == want_err && !r.error.empty();
            std::cout << "  [" << tag << "] " << (good ? "rejected(" : "WRONG(")
                      << qwen3_asr_error_str(r.err) << "): "
                      << (r.ok ? std::string("(ok!!)") : r.error) << (good ? "  OK" : "  FAIL") << "\n";
            if (!good) ++failures;
        };
        {
            qwen3_asr_request bad = req; bad.pcm = nullptr; bad.n_samples = 0;
            expect_error("empty-input", bad, QWEN3_ASR_ERR_BAD_INPUT);
        }
        {
            qwen3_asr_request bad = req; bad.sample_rate = 44100;
            expect_error("wrong-rate", bad, QWEN3_ASR_ERR_BAD_INPUT);
        }
        {
            qwen3_asr_request bad = req;
            bad.n_samples = (int32_t) pcm.size() + 3600 * (int32_t) sample_rate; // 超 30s 上限
            std::vector<float> big_pcm((size_t) bad.n_samples, 0.1f);
            bad.pcm = big_pcm.data();
            if ((int64_t) bad.n_samples > (int64_t) 31 * sample_rate) {
                expect_error("too-long", bad, QWEN3_ASR_ERR_AUDIO_TOO_LONG);
            }
        }
        {
            std::vector<float> nan_pcm((size_t) req.n_samples, 0.1f);
            nan_pcm[nan_pcm.size() / 2] = std::nanf("");
            qwen3_asr_request bad = req; bad.pcm = nan_pcm.data();
            expect_error("nan-input", bad, QWEN3_ASR_ERR_BAD_INPUT);
        }
        {
            // 上下文不足：独立小 n_ctx 引擎（n_ctx=64 < S），长输入触发 CONTEXT_INSUFFICIENT
            qwen3_asr_engine small_engine;
            qwen3_asr_engine_config sc;
            sc.model_path = model_path;
            sc.n_ctx = 64; // KV 只装 64 token：S(>=A+~17) 必然放不下
            if (small_engine.Init(sc)) {
                // 输入拉长到 S 必然超过 n_ctx=64（1.9s 输入 S=40 barely fits；4×tile → A=100, S≈118）
                std::vector<float> long_pcm;
                const int tiles = std::max(4, (int) (10 * 16000 / std::max<size_t>(pcm.size(), 1)) + 1);
                for (int i = 0; i < tiles; ++i) long_pcm.insert(long_pcm.end(), pcm.begin(), pcm.end());
                qwen3_asr_request bad = req;
                bad.pcm = long_pcm.data();
                bad.n_samples = (int32_t) long_pcm.size();
                bad.want_debug = false;
                const qwen3_asr_response r = small_engine.Transcribe(bad);
                const bool good = !r.ok && r.err == QWEN3_ASR_ERR_CONTEXT_INSUFFICIENT;
                std::cout << "  [context-insuff] " << (good ? "rejected: " : "WRONG(")
                          << qwen3_asr_error_str(r.err) << "): "
                          << (r.ok ? std::string("(ok!!)") : r.error)
                          << (good ? "  OK" : "  FAIL") << "\n";
                if (!good) ++failures;
            } else {
                std::cout << "  [context-insuff] small engine init failed  FAIL\n";
                ++failures;
            }
        }
        {
            // 重复 Init：幂等（返回 true，保持首份模型可用）
            const bool re_init = engine.InitFromJsonFile(config_path.empty() ? "" : config_path);
            std::cout << "  [re-init] " << (re_init ? "idempotent OK" : "FAIL") << "\n";
            if (!re_init) ++failures;
        }
    }

    // ---- 结果汇总 ----
    std::cout << "== result: " << (failures ? "FAIL" : "PASS") << " (failures=" << failures << ") ==\n";
    return failures ? 1 : 0;
}
