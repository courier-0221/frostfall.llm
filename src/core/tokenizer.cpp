#include "tokenizer.h"

#include "gguf.h"

#include "log.h"

#include <algorithm>
#include <array>

// =============================================================================
// Qwen3 分词器实现（byte-level BPE）
//
// 处理流程（encode）：
//   text
//    └─(1) 按特殊 token 字面量切分（<|im_start|> 等直接给 id，不参与 BPE）
//         └─(2) pretokenize：Qwen2 正则把普通文本切成“词片段”
//              └─(3) byte encode：每个片段的 UTF-8 字节逐个映射成 GPT-2 的可打印字符
//                   └─(4) BPE：按 merges 优先级把相邻片段合并成词表里的 token
//                        └─ token 字符串 -> id
//
// decode 是逆过程：id -> 字节级编码字符串 -> 逆映射回原始字节 -> 拼成 UTF-8 文本。
// =============================================================================

namespace {

// ---------------------------------------------------------------------------
// UTF-8 <-> Unicode 码点
// ---------------------------------------------------------------------------

// 把一段 UTF-8 字节流解码成码点数组（非法字节按单字节 0xFFFD 容错处理）。
std::vector<uint32_t> utf8_to_cps(const std::string & s) {
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const uint8_t c = (uint8_t) s[i];
        uint32_t cp = 0;
        int extra = 0;
        if (c < 0x80)        { cp = c;        extra = 0; }
        else if (c < 0xE0)   { cp = c & 0x1F; extra = 1; }
        else if (c < 0xF0)   { cp = c & 0x0F; extra = 2; }
        else                 { cp = c & 0x07; extra = 3; }
        if (i + extra >= n) { cps.push_back(0xFFFD); break; }
        bool ok = true;
        for (int k = 1; k <= extra; ++k) {
            const uint8_t cc = (uint8_t) s[i + k];
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { cps.push_back(0xFFFD); i += 1; continue; }
        cps.push_back(cp);
        i += extra + 1;
    }
    return cps;
}

// 把一个码点追加编码成 UTF-8 字节到 out。
void cp_to_utf8(uint32_t cp, std::string & out) {
    if (cp < 0x80) {
        out.push_back((char) cp);
    } else if (cp < 0x800) {
        out.push_back((char) (0xC0 | (cp >> 6)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char) (0xE0 | (cp >> 12)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char) (0xF0 | (cp >> 18)));
        out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    }
}

// ---------------------------------------------------------------------------
// GPT-2 bytes_to_unicode：把 0..255 每个字节映射成一个“可打印”的 Unicode 码点。
// 目的：让任意字节都能安全地当作文本参与 BPE（词表里存的就是这些映射后的字符）。
// 规则：可打印区间 [33,126]∪[161,172]∪[174,255] 的字节映射到自身码点，
//       其余字节按出现顺序映射到 256,257,258...（例如空格 0x20 -> 0x120 = 'Ġ'）。
// ---------------------------------------------------------------------------
struct byte_coder {
    std::array<uint32_t, 256> byte_to_cp{};      // 字节 -> 码点
    std::unordered_map<uint32_t, uint8_t> cp_to_byte; // 码点 -> 字节（decode 用）

    byte_coder() {
        auto printable = [](int b) {
            return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        };
        uint32_t n = 0;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp;
            if (printable(b)) {
                cp = (uint32_t) b;
            } else {
                cp = 256 + n;
                ++n;
            }
            byte_to_cp[b] = cp;
            cp_to_byte[cp] = (uint8_t) b;
        }
    }
};

const byte_coder & coder() {
    static const byte_coder c;
    return c;
}

// 把一段原始字节（一个词片段）编码成“字节级字符”串：每个字节 -> 对应码点 -> UTF-8。
std::string byte_encode(const std::string & piece) {
    std::string out;
    out.reserve(piece.size() * 2);
    for (unsigned char b : piece) {
        cp_to_utf8(coder().byte_to_cp[b], out);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Unicode 字符分类（Qwen2 pretokenizer 正则用到 \p{L} \p{N} \s）。
// 说明：这里用码点区间近似 Unicode 属性，覆盖英文/中文/日文/韩文/常见欧洲语言，
//       足够跑通教学用例；不追求对所有脚本 100% 精确（那需要完整 Unicode 表）。
// ---------------------------------------------------------------------------
bool is_whitespace(uint32_t c) {
    switch (c) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
        case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return (c >= 0x2000 && c <= 0x200A);
    }
}

bool is_number(uint32_t c) { // \p{Nd} 近似
    return (c >= '0' && c <= '9') ||
           (c >= 0x0660 && c <= 0x0669) ||
           (c >= 0x06F0 && c <= 0x06F9) ||
           (c >= 0xFF10 && c <= 0xFF19);
}

bool is_letter(uint32_t c) { // \p{L} 近似
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return true;
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
    if (c >= 0x00C0 && c <= 0x024F) return c != 0x00D7 && c != 0x00F7; // 拉丁扩展（排除 × ÷）
    if (c >= 0x0370 && c <= 0x03FF) return true;                       // 希腊
    if (c >= 0x0400 && c <= 0x04FF) return true;                       // 西里尔
    if (c >= 0x0531 && c <= 0x058F) return true;                       // 亚美尼亚
    if (c >= 0x05D0 && c <= 0x05EA) return true;                       // 希伯来
    if (c >= 0x0620 && c <= 0x064A) return true;                       // 阿拉伯字母
    if (c >= 0x066E && c <= 0x06D3) return true;
    if (c >= 0x0E00 && c <= 0x0E7F) return true;                       // 泰文
    if (c >= 0x1100 && c <= 0x11FF) return true;                       // 韩文字母
    if (c >= 0x3040 && c <= 0x30FF) return true;                       // 平假名/片假名
    if (c >= 0x3400 && c <= 0x4DBF) return true;                       // CJK 扩展 A
    if (c >= 0x4E00 && c <= 0x9FFF) return true;                       // CJK 基本
    if (c >= 0xAC00 && c <= 0xD7A3) return true;                       // 韩文音节
    if (c >= 0xF900 && c <= 0xFAFF) return true;                       // CJK 兼容
    if (c >= 0xFF21 && c <= 0xFF3A) return true;                       // 全角大写字母
    if (c >= 0xFF41 && c <= 0xFF5A) return true;                       // 全角小写字母
    if (c >= 0x20000 && c <= 0x2FA1F) return true;                     // CJK 扩展 B~
    return false;
}

inline uint32_t ascii_lower(uint32_t c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c;
}

// ---------------------------------------------------------------------------
// pretokenize：复刻 Qwen2 的正则（llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2）：
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
// 在每个位置按顺序尝试上述分支，取第一个匹配的作为一个词片段。
// 这里不用 std::regex（不支持 \p{L} 且回溯语义难对齐），而是手写等价扫描器。
// ---------------------------------------------------------------------------
std::vector<std::string> pretokenize(const std::string & text) {
    const std::vector<uint32_t> cps = utf8_to_cps(text);
    const int n = (int) cps.size();

    std::vector<std::string> pieces;
    auto emit = [&](int a, int b) {
        std::string s;
        for (int j = a; j < b; ++j) cp_to_utf8(cps[j], s);
        pieces.push_back(std::move(s));
    };

    int i = 0;
    while (i < n) {
        // 分支 1：英文缩写 's 't 're 've 'm 'll 'd（大小写不敏感）
        if (cps[i] == '\'' && i + 1 < n) {
            const uint32_t c1 = ascii_lower(cps[i + 1]);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { emit(i, i + 2); i += 2; continue; }
            if (i + 2 < n) {
                const uint32_t c2 = ascii_lower(cps[i + 2]);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    emit(i, i + 3); i += 3; continue;
                }
            }
        }

        // 分支 2：[^\r\n\p{L}\p{N}]? \p{L}+   —— 可选一个前导非字母数字符（含前导空格），再接一串字母
        {
            int j = i;
            const bool lead_ok = cps[j] != '\r' && cps[j] != '\n' && !is_letter(cps[j]) && !is_number(cps[j]);
            if (lead_ok && j + 1 < n && is_letter(cps[j + 1])) {
                j++; // 吃掉前导字符（例如把 " word" 的空格并进来）
            }
            if (j < n && is_letter(cps[j])) {
                while (j < n && is_letter(cps[j])) j++;
                emit(i, j); i = j; continue;
            }
        }

        // 分支 3：\p{N}  —— 单个数字自成一个片段（Qwen2 特性：数字逐位切分）
        if (is_number(cps[i])) { emit(i, i + 1); i++; continue; }

        // 分支 4： ?[^\s\p{L}\p{N}]+[\r\n]*  —— 可选前导空格 + 一串标点/符号 + 尾随换行
        {
            int j = i;
            auto is_punct = [&](uint32_t c) { return !is_whitespace(c) && !is_letter(c) && !is_number(c); };
            if (cps[j] == ' ' && j + 1 < n && is_punct(cps[j + 1])) j++;
            if (j < n && is_punct(cps[j])) {
                while (j < n && is_punct(cps[j])) j++;
                while (j < n && (cps[j] == '\r' || cps[j] == '\n')) j++;
                emit(i, j); i = j; continue;
            }
        }

        // 分支 5/6/7：剩下的都是空白。整段空白一次性处理。
        if (is_whitespace(cps[i])) {
            int we = i;
            while (we < n && is_whitespace(cps[we])) we++;

            // 分支 5：\s*[\r\n]+ —— 若空白段里含换行，切到最后一个换行为止
            int last_nl = -1;
            for (int j = i; j < we; ++j) if (cps[j] == '\n' || cps[j] == '\r') last_nl = j;
            if (last_nl >= 0) { emit(i, last_nl + 1); i = last_nl + 1; continue; }

            // 分支 6：\s+(?!\S) —— 空白段后面若还跟着非空白（下一个词），留一个空格给它
            if (we < n && we - i >= 2) { emit(i, we - 1); i = we - 1; continue; }

            // 分支 7：\s+ —— 其余空白（段尾空白、或单个空格）
            emit(i, we); i = we; continue;
        }

        // 理论上到不了这里；兜底吃一个码点，避免死循环
        emit(i, i + 1); i++;
    }

    return pieces;
}

} // namespace

// ---------------------------------------------------------------------------
// 加载：从 GGUF 读词表 / merges / token_type / 特殊 token id
// ---------------------------------------------------------------------------
bool qwen3_tokenizer::load(const std::string & fname) {
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        LOG(ERROR) << __func__ << ": gguf_init_from_file() failed for '" << fname << "'";
        return false;
    }

    const int64_t kid_tokens = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    const int64_t kid_types  = gguf_find_key(ctx, "tokenizer.ggml.token_type");
    const int64_t kid_merges = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (kid_tokens < 0 || kid_merges < 0) {
        LOG(ERROR) << __func__ << ": GGUF is missing tokenizer.ggml.tokens / merges";
        gguf_free(ctx);
        return false;
    }

    const size_t n_tokens = gguf_get_arr_n(ctx, kid_tokens);
    id_to_token.resize(n_tokens);
    token_type.assign(n_tokens, 1 /*NORMAL*/);
    token_to_id.reserve(n_tokens * 2);

    const int32_t * types = kid_types >= 0 ? (const int32_t *) gguf_get_arr_data(ctx, kid_types) : nullptr;

    for (size_t i = 0; i < n_tokens; ++i) {
        const char * t = gguf_get_arr_str(ctx, kid_tokens, i);
        id_to_token[i] = t;
        token_to_id[t] = (int32_t) i;
        if (types) token_type[i] = types[i];
        if (token_type[i] == 3 /*CONTROL*/ || token_type[i] == 4 /*USER_DEFINED*/) {
            special_tokens.emplace_back(id_to_token[i], (int32_t) i);
        }
    }

    const size_t n_merges = gguf_get_arr_n(ctx, kid_merges);
    merge_rank.reserve(n_merges * 2);
    for (size_t i = 0; i < n_merges; ++i) {
        // 合并规则形如 "左 右"（左右两片段本身不含空格字节，故用第一个空格分隔无歧义）。
        merge_rank[gguf_get_arr_str(ctx, kid_merges, i)] = (int32_t) i;
    }

    // 特殊 token 按字面量长度从长到短排序，encode 时优先匹配更长的（避免前缀误伤）。
    std::sort(special_tokens.begin(), special_tokens.end(),
              [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });

    auto get_u32 = [&](const char * key, int32_t def) -> int32_t {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) return def;
        switch (gguf_get_kv_type(ctx, kid)) {
            case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(ctx, kid);
            case GGUF_TYPE_INT32:  return           gguf_get_val_i32(ctx, kid);
            default:               return def;
        }
    };
    eos_id = get_u32("tokenizer.ggml.eos_token_id", -1);
    bos_id = get_u32("tokenizer.ggml.bos_token_id", -1);
    pad_id = get_u32("tokenizer.ggml.padding_token_id", -1);

    const int64_t kid_add_bos = gguf_find_key(ctx, "tokenizer.ggml.add_bos_token");
    if (kid_add_bos >= 0 && gguf_get_kv_type(ctx, kid_add_bos) == GGUF_TYPE_BOOL) {
        add_bos = gguf_get_val_bool(ctx, kid_add_bos);
    }

    LOG(INFO) << __func__ << ": vocab=" << n_tokens << " merges=" << n_merges
              << " special=" << special_tokens.size()
              << " eos=" << eos_id << " bos=" << bos_id << " add_bos=" << (add_bos ? "true" : "false");

    gguf_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// BPE：把一个“字节级编码字符串”按 merges 优先级合并成词表 token，追加它们的 id 到 out。
// ---------------------------------------------------------------------------
namespace {
void bpe_encode_piece(const qwen3_tokenizer & tk, const std::string & encoded, std::vector<int32_t> & out) {
    // 初始 symbols：每个 symbol 是“一个码点”的 UTF-8 串
    std::vector<std::string> symbols;
    {
        const std::vector<uint32_t> cps = utf8_to_cps(encoded);
        symbols.reserve(cps.size());
        for (uint32_t cp : cps) { std::string s; cp_to_utf8(cp, s); symbols.push_back(std::move(s)); }
    }
    if (symbols.empty()) return;

    // 反复合并优先级最高（rank 最小）的相邻 pair，一次把该 pair 的所有出现都合并（GPT-2 BPE 语义）。
    while (symbols.size() > 1) {
        int best_rank = INT32_MAX;
        int best_i = -1;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = tk.merge_rank.find(symbols[i] + " " + symbols[i + 1]);
            if (it != tk.merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = (int) i;
            }
        }
        if (best_i < 0) break; // 没有可合并的 pair

        const std::string & first  = symbols[best_i];
        const std::string   merged = first + symbols[best_i + 1];
        std::vector<std::string> next;
        next.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size();) {
            if ((int) i == best_i) { next.push_back(merged); i += 2; }
            else                   { next.push_back(symbols[i]); i += 1; }
            // 说明：一轮只合并 best pair 的“最左一次”出现即可保证收敛且与参考实现等价，
            //       因为下一轮会再次选中同一 pair 继续合并剩余出现。为清晰起见这里逐次合并。
        }
        symbols.swap(next);
    }

    for (const std::string & sym : symbols) {
        auto it = tk.token_to_id.find(sym);
        if (it != tk.token_to_id.end()) {
            out.push_back(it->second);
        } else {
            // 理论上不会发生：所有单字节编码字符都在词表里。真出现则逐字符兜底。
            LOG(WARNING) << "bpe: token not in vocab: '" << sym << "'";
            const std::vector<uint32_t> cps = utf8_to_cps(sym);
            for (uint32_t cp : cps) {
                std::string s; cp_to_utf8(cp, s);
                auto j = tk.token_to_id.find(s);
                if (j != tk.token_to_id.end()) out.push_back(j->second);
            }
        }
    }
}
} // namespace

std::vector<int32_t> qwen3_tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    if (add_bos && bos_id >= 0) ids.push_back(bos_id);

    // 先按特殊 token 字面量把文本切成 [普通段 | 特殊 token | 普通段 | ...]
    size_t pos = 0;
    while (pos < text.size()) {
        // 在当前位置尝试匹配任一特殊 token（special_tokens 已按长度降序，优先长匹配）
        int32_t matched_id = -1;
        size_t  matched_len = 0;
        for (const auto & sp : special_tokens) {
            const std::string & lit = sp.first;
            if (!lit.empty() && text.compare(pos, lit.size(), lit) == 0) {
                matched_id = sp.second;
                matched_len = lit.size();
                break;
            }
        }

        if (matched_id >= 0) {
            ids.push_back(matched_id);
            pos += matched_len;
            continue;
        }

        // 找到下一个特殊 token 的起点，把 [pos, next) 作为普通文本处理
        size_t next = text.size();
        for (const auto & sp : special_tokens) {
            const size_t f = text.find(sp.first, pos);
            if (f != std::string::npos) next = std::min(next, f);
        }

        const std::string chunk = text.substr(pos, next - pos);
        for (const std::string & piece : pretokenize(chunk)) {
            bpe_encode_piece(*this, byte_encode(piece), ids);
        }
        pos = next;
    }

    return ids;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
std::string qwen3_tokenizer::id_to_piece(int32_t id, bool skip_special) const {
    if (id < 0 || id >= (int32_t) id_to_token.size()) return "";
    if (is_special(id)) {
        return skip_special ? std::string() : id_to_token[id];
    }
    // NORMAL：token 字符串是“字节级编码”，逐码点逆映射回原始字节。
    std::string out;
    for (uint32_t cp : utf8_to_cps(id_to_token[id])) {
        auto it = coder().cp_to_byte.find(cp);
        if (it != coder().cp_to_byte.end()) out.push_back((char) it->second);
    }
    return out;
}

std::string qwen3_tokenizer::decode(const std::vector<int32_t> & ids, bool skip_special) const {
    std::string out;
    for (int32_t id : ids) out += id_to_piece(id, skip_special);
    return out;
}
