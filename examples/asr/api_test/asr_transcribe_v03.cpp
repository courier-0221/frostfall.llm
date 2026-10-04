// ============================================================================
// asr_transcribe_v03.cpp —— asr/v0.3 纯 C++ 离线转写 + 全链路对齐示例
//
// 相比 v0.2（输入参考 Mel，只验证音频塔 + Decoder 主体），v0.3 用最小阻塞
// API（src/asr_engine.*）直接输入 PCM：
//   WAV/PCM16 → /32768 → float32 16k PCM → AsrEngine::Transcribe → 语言 + 文字
// 不再需要 Python 运行时；Log-Mel（src/log_mel.*）在引擎内部完成。
//
// 三种运行模式：
//   1. 离线转写（默认）：读 WAV（单声道 16 kHz PCM16），Transcribe 两次并
//      校验第二次结果与第一次一致（验证每请求 KV 重置、重复调用不串状态）；
//   2. 对齐模式（--ref-dir）：用参考目录的 pcm.npy 作为输入，开启
//      want_debug 对比 Log-Mel vs golden mel.npy，生成 ids vs golden
//      decode_ids.npy，验收标准见 doc/asr/design_asr.md §4.5；
//   3. 错误演示（--check-errors/embedded）：空输入 / 非 16 kHz / 超长 /
//      NaN 逐一调用，验证显式报错、不静默截断。
//
// 用法：
//   asr_transcribe_v03 --config examples/asr/api_test/asr_conf_v03.json \
//                      --wav data/test_asr_001.wav
//   asr_transcribe_v03 --config ... --ref-dir work/asr_ref_v02/auto
//   asr_transcribe_v03 --config ... --ref-dir work/asr_ref_v02/lang --language Chinese
//   asr_transcribe_v03 ... --wav x.wav --context "上下文提示" --max-gen 96
// ============================================================================

#include "asr_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ============================ .npy 读取（同 v0.1/v0.02 对齐工具的最小实现） ============================

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

// ============================ PCM16 WAV 读取（单声道 16 kHz，/32768） ============================

struct WavInfo {
    int32_t sample_rate = 0;
    int32_t channels    = 0;
    int32_t bits        = 0;
    int32_t format      = 0;
};

// 只接受 RIFF/WAVE、非压缩 PCM（format=1）、16-bit；声道/采样率由 engine 侧校验。
static bool read_wav_pcm16(const std::string & path, std::vector<float> & pcm_out,
                           WavInfo & info, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open wav: " + path; return false; }

    char hdr[12];
    if (!f.read(hdr, 12) || std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file: " + path;
        return false;
    }
    bool have_fmt = false, have_data = false;
    while (f && !have_data) {
        char id[4]; uint32_t sz = 0;
        if (!f.read(id, 4)) break;
        f.read(reinterpret_cast<char *>(&sz), 4);
        const std::streamoff next_off = (std::streamoff) f.tellg()
                                        + (std::streamoff) (((sz + 1) / 2) * 2); // chunk 按 2 字节对齐
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t audio_format = 0, channels = 0, bits = 0;
            uint32_t rate = 0;
            f.read(reinterpret_cast<char *>(&audio_format), 2);
            f.read(reinterpret_cast<char *>(&channels), 2);
            f.read(reinterpret_cast<char *>(&rate), 4);
            f.seekg(6, std::ios::cur); // byte_rate(4) + block_align(2)
            f.read(reinterpret_cast<char *>(&bits), 2);
            info.format = audio_format;
            info.channels = channels;
            info.sample_rate = (int32_t) rate;
            info.bits = bits;
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!have_fmt) { err = "wav 'data' chunk before 'fmt '"; return false; }
            if (info.format != 1) {
                err = "unsupported wav audio format " + std::to_string(info.format) +
                      " (only uncompressed PCM=1; float/other codecs must be converted first)";
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
            for (size_t i = 0; i < raw.size(); ++i)
                pcm_out[i] = (float) raw[i] / 32768.0f; // PCM16 -> float32，与参考导出一致
            have_data = true;
        }
        f.clear(); // data 读完或未知 chunk：跳到下一个 chunk 起点
        f.seekg(next_off, std::ios::beg);
    }
    if (!have_fmt || !have_data) { err = "incomplete wav (missing fmt/data chunk): " + path; return false; }
    return true;
}

// ============================ 打印 ============================

static void print_response(const std::string & tag, const qwen3_asr_response & r, bool debug) {
    if (!r.ok) {
        std::cout << "[" << tag << "] ERROR: " << r.error << "\n";
        return;
    }
    std::cout << "[" << tag << "] finish=" << qwen3_asr_finish_reason_str(r.finish_reason)
              << " language=" << (r.language.empty() ? "<none>" : r.language)
              << " text=" << r.text << "\n";
    if (debug) {
        std::cout << "  [debug] T=" << r.T << " A=" << r.A << " S=" << r.S
                  << " n_gen=" << r.n_gen << " log_mel_max=" << r.log_mel_max << "\n";
        if (!r.raw.empty()) std::cout << "  [raw ] " << r.raw << "\n";
    }
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

// ============================ main ============================

int main(int argc, char ** argv) {
    std::string config_path, gguf_path, wav_path, pcm_path, ref_dir, language, context;
    int32_t max_gen = 0;
    bool check_errors = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (a == "--gguf"   && i + 1 < argc) gguf_path   = argv[++i];
        else if (a == "--wav"    && i + 1 < argc) wav_path    = argv[++i];
        else if (a == "--pcm"    && i + 1 < argc) pcm_path    = argv[++i]; // float32 npy（16k）
        else if (a == "--ref-dir" && i + 1 < argc) ref_dir    = argv[++i];
        else if (a == "--language" && i + 1 < argc) language  = argv[++i];
        else if (a == "--context"  && i + 1 < argc) context   = argv[++i];
        else if (a == "--max-gen"  && i + 1 < argc) max_gen   = std::atoi(argv[++i]);
        else if (a == "--check-errors") check_errors = true;
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " --config <asr_conf_v03.json> [--gguf <model.gguf>]\n"
                      << "        [--wav <in.wav> | --pcm <pcm.npy> | --ref-dir <dir>]\n"
                      << "        [--language Chinese] [--context str] [--max-gen N] [--check-errors]\n";
            return 0;
        }
    }
    if (config_path.empty() && gguf_path.empty()) {
        std::cerr << "missing --config or --gguf\n";
        return 1;
    }
    if (wav_path.empty() && pcm_path.empty() && ref_dir.empty()) {
        std::cerr << "missing input: one of --wav / --pcm / --ref-dir required\n";
        return 1;
    }

    std::cout << "== asr/v0.3 offline transcribe ==\n";

    // ---- 引擎初始化 ----
    qwen3_asr_engine engine;
    qwen3_asr_engine_config cfg;
    if (!config_path.empty()) {
        std::cout << "engine init from " << config_path << "\n";
        if (!engine.InitFromJsonFile(config_path)) { std::cerr << "InitFromJsonFile failed\n"; return 1; }
    } else {
        cfg.model_path = gguf_path;
        std::cout << "engine init from " << gguf_path << "\n";
        if (!engine.Init(cfg)) { std::cerr << "Init failed\n"; return 1; }
    }
    std::cout << "engine ready (n_ctx=" << engine.NCtx() << ")\n";

    int failures = 0;

    // ---- 输入 PCM ----
    std::vector<float> pcm;
    int32_t sample_rate = 16000;
    if (!ref_dir.empty()) {
        try {
            const NpyArray p = load_npy(ref_dir + "/pcm.npy");
            pcm.assign(p.as_f32(), p.as_f32() + p.count());
            sample_rate = 16000;
        } catch (const std::exception & e) { std::cerr << e.what() << "\n"; return 1; }
        std::cout << "input: ref pcm " << pcm.size() << " samples (" << pcm.size() / 16000.0 << "s)\n";
    } else if (!wav_path.empty()) {
        WavInfo info;
        std::string werr;
        if (!read_wav_pcm16(wav_path, pcm, info, werr)) { std::cerr << werr << "\n"; return 1; }
        sample_rate = info.sample_rate;
        std::cout << "input: wav " << pcm.size() << " samples, " << info.sample_rate << " Hz, "
                  << info.channels << "ch, " << info.bits << "bit\n";
    } else {
        try {
            const NpyArray p = load_npy(pcm_path);
            pcm.assign(p.as_f32(), p.as_f32() + p.count());
        } catch (const std::exception & e) { std::cerr << e.what() << "\n"; return 1; }
        std::cout << "input: pcm " << pcm.size() << " samples\n";
    }

    // ---- 转写（想看 debug：对齐模式必须开；普通模式也默认开便于观察） ----
    qwen3_asr_request req;
    req.pcm = pcm.data();
    req.n_samples = (int32_t) pcm.size();
    req.sample_rate = sample_rate;
    req.language = language;
    req.context = context;
    req.max_gen_tokens = max_gen;
    req.want_debug = true;

    const qwen3_asr_response r1 = engine.Transcribe(req);
    print_response("transcribe#1", r1, /*debug=*/true);
    if (!r1.ok) return 1;

    // ---- 重复调用一致性（每请求 KV 重置，不串状态） ----
    const qwen3_asr_response r2 = engine.Transcribe(req);
    print_response("transcribe#2", r2, /*debug=*/false);
    if (!r2.ok) {
        std::cout << "  [repeat] Second call failed: " << r2.error << "  FAIL\n";
        ++failures;
    } else if (r2.text != r1.text || r2.language != r1.language ||
               r2.finish_reason != r1.finish_reason || r2.gen_ids != r1.gen_ids) {
        std::cout << "  [repeat] Second call differs from first  FAIL\n";
        ++failures;
    } else {
        std::cout << "  [repeat] identical to first call  OK\n";
    }

    // ---- 参考对齐（--ref-dir）：Log-Mel 与 golden 逐值对比、生成 ids 对比 ----
    if (!ref_dir.empty()) {
        std::cout << "-- golden alignment (" << ref_dir << ") --\n";
        try {
            const NpyArray g_mel = load_npy(ref_dir + "/mel.npy");        // [n_mels, T]
            const NpyArray g_ids = load_npy(ref_dir + "/decode_ids.npy"); // [n_gen]
            if (g_mel.shape[0] != (size_t) 128 || g_mel.shape[1] != (size_t) r1.T) {
                std::cout << "  [mel] shape mismatch: golden [" << g_mel.shape[0] << ","
                          << g_mel.shape[1] << "] vs cpp T=" << r1.T << "  FAIL\n";
                ++failures;
            } else {
                double rmse = 0.0;
                const double mx = max_abs_diff(r1.mel.data(), g_mel.as_f32(),
                                               (size_t) 128 * r1.T, &rmse);
                std::cout << "  [log_mel] n=" << (size_t) 128 * r1.T
                          << " max_abs=" << mx << " rmse=" << rmse
                          << ((mx < 1e-4 && rmse < 1e-5) ? "  OK" : "  FAIL") << "\n";
                if (!(mx < 1e-4 && rmse < 1e-5)) ++failures;
            }
            {
                const int32_t n_ref = (int32_t) g_ids.shape[0];
                const int32_t n_cmn = std::min<int32_t>(n_ref, (int32_t) r1.gen_ids.size());
                int match = 0;
                for (int32_t i = 0; i < n_cmn; ++i)
                    match += (r1.gen_ids[(size_t) i] == g_ids.as_i32()[i]) ? 1 : 0;
                std::cout << "  [gen_ids] n_cpp=" << r1.gen_ids.size() << " n_ref=" << n_ref
                          << " prefix_match=" << match << "/" << n_cmn;
                if ((int32_t) r1.gen_ids.size() == n_ref && match == n_cmn) {
                    std::cout << "  OK\n";
                } else {
                    std::cout << "  FAIL\n";
                    ++failures;
                }
            }
        } catch (const std::exception & e) {
            std::cout << "  [golden] " << e.what() << "  FAIL\n";
            ++failures;
        }
    }

    // ---- 附加：非空 context 演示（验证请求字段生效路径） ----
    if (ref_dir.empty() && language.empty()) {
        qwen3_asr_request req_ctx = req;
        req_ctx.context = "转写时保留标点。";
        const qwen3_asr_response rc = engine.Transcribe(req_ctx);
        std::cout << "[with-context] ok=" << rc.ok;
        if (rc.ok) std::cout << " finish=" << qwen3_asr_finish_reason_str(rc.finish_reason)
                             << " text=" << rc.text;
        else        std::cout << " error=" << rc.error;
        std::cout << "\n";
    }

    // ---- 错误路径演示（--check-errors）：显式报错、不静默截断 ----
    if (check_errors) {
        std::cout << "-- error checks --\n";
        auto expect_error = [&](const std::string & tag, const qwen3_asr_request & bad_req) {
            const qwen3_asr_response r = engine.Transcribe(bad_req);
            const bool good = !r.ok && !r.error.empty();
            std::cout << "  [" << tag << "] " << (good ? "rejected: " : "NOT rejected: ")
                      << (r.ok ? "(ok!!)" : r.error) << (good ? "  OK" : "  FAIL") << "\n";
            if (!good) ++failures;
        };
        qwen3_asr_request bad = req;
        bad.pcm = nullptr;
        bad.n_samples = 0;
        expect_error("empty-input", bad);

        bad = req;
        bad.sample_rate = 44100;
        expect_error("wrong-rate", bad);

        bad = req;
        bad.n_samples = 9 * 16000; // 超过 max_audio_seconds=8（不读入，只改长度）
        expect_error("too-long", bad);

        std::vector<float> nan_pcm = pcm;
        if (!nan_pcm.empty()) nan_pcm[nan_pcm.size() / 2] = std::nanf("");
        bad = req;
        bad.pcm = nan_pcm.data();
        bad.n_samples = (int32_t) nan_pcm.size();
        expect_error("nan-input", bad);
    }

    std::cout << "== result: " << (failures == 0 ? "PASS" : "FAIL")
              << " (failures=" << failures << ") ==\n";
    return failures == 0 ? 0 : 1;
}
