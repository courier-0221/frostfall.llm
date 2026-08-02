// Qwen3 聊天协议编解码实现（声明见 qwen3_chat.h）
//
// 不依赖 JSON 库；用花括号/引号匹配手写解析与容错规整。

#include "qwen3_chat.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace Frostfall {

// ============================================================
// JSON 容错规整器（仅本文件内部使用）
//
// 把模型输出的（可能降级的）arguments 字面量修复成合法紧凑 JSON：
//   {"name": "ACControl", "arguments": {"action":"open","temperature":24}}
// ============================================================
namespace {

void json_skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
        ++i;
    }
}

// 把裸 token 作为 JSON 字符串输出（转义必要字符）
void json_append_quoted(std::string& out, const std::string& token) {
    out.push_back('"');
    for (char c : token) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out.push_back(c); break;
        }
    }
    out.push_back('"');
}

// 判断裸 token 是否为合法 JSON 基础类型（数字 / true / false / null），
// 是则原样保留，否则需要加引号当字符串。
bool json_is_primitive(const std::string& t) {
    if (t == "true" || t == "false" || t == "null") return true;
    if (t.empty()) return false;
    size_t i = 0;
    if (t[i] == '-' || t[i] == '+') ++i;
    bool has_digit = false, has_dot = false, has_exp = false;
    for (; i < t.size(); ++i) {
        char c = t[i];
        if (c >= '0' && c <= '9') {
            has_digit = true;
        } else if (c == '.' && !has_dot && !has_exp) {
            has_dot = true;
        } else if ((c == 'e' || c == 'E') && !has_exp && has_digit) {
            has_exp = true;
            if (i + 1 < t.size() && (t[i + 1] == '+' || t[i + 1] == '-')) ++i;
        } else {
            return false;
        }
    }
    return has_digit;
}

// 拷贝一个带引号的字符串字面量（保留内部转义），i 指向起始的 '"'
void json_copy_string(const std::string& s, size_t& i, std::string& out) {
    out.push_back('"');
    ++i;  // 跳过开引号
    bool esc = false;
    while (i < s.size()) {
        char c = s[i++];
        if (esc) {
            out.push_back(c);
            esc = false;
        } else if (c == '\\') {
            out.push_back(c);
            esc = true;
        } else if (c == '"') {
            break;
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
}

void json_normalize_value(const std::string& s, size_t& i, std::string& out);

void json_normalize_object(const std::string& s, size_t& i, std::string& out) {
    out.push_back('{');
    ++i;  // 跳过 '{'
    json_skip_ws(s, i);
    bool first = true;
    while (i < s.size() && s[i] != '}') {
        if (!first) out.push_back(',');
        first = false;

        // ---- key ----
        json_skip_ws(s, i);
        if (i < s.size() && s[i] == '"') {
            json_copy_string(s, i, out);
        } else {
            std::string key;
            while (i < s.size() && s[i] != ':' && s[i] != '=' && s[i] != ',' &&
                   s[i] != '}' && s[i] != ' ' && s[i] != '\t' && s[i] != '\n' &&
                   s[i] != '\r') {
                key.push_back(s[i++]);
            }
            json_append_quoted(out, key);
        }

        // ---- 分隔符 ':' 或 '=' ----
        json_skip_ws(s, i);
        if (i < s.size() && (s[i] == ':' || s[i] == '=')) ++i;
        out.push_back(':');

        // ---- value ----
        json_normalize_value(s, i, out);

        json_skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            json_skip_ws(s, i);
        }
    }
    if (i < s.size() && s[i] == '}') ++i;
    out.push_back('}');
}

void json_normalize_array(const std::string& s, size_t& i, std::string& out) {
    out.push_back('[');
    ++i;  // 跳过 '['
    json_skip_ws(s, i);
    bool first = true;
    while (i < s.size() && s[i] != ']') {
        if (!first) out.push_back(',');
        first = false;
        json_normalize_value(s, i, out);
        json_skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            json_skip_ws(s, i);
        }
    }
    if (i < s.size() && s[i] == ']') ++i;
    out.push_back(']');
}

void json_normalize_value(const std::string& s, size_t& i, std::string& out) {
    json_skip_ws(s, i);
    if (i >= s.size()) {
        out += "\"\"";
        return;
    }
    char c = s[i];
    if (c == '{') {
        json_normalize_object(s, i, out);
    } else if (c == '[') {
        json_normalize_array(s, i, out);
    } else if (c == '"') {
        json_copy_string(s, i, out);
    } else {
        // 裸值：扫到 , } ] 为止
        std::string token;
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') {
            token.push_back(s[i++]);
        }
        while (!token.empty() &&
               (token.back() == ' ' || token.back() == '\t' ||
                token.back() == '\n' || token.back() == '\r')) {
            token.pop_back();
        }
        if (json_is_primitive(token)) {
            out += token;
        } else {
            json_append_quoted(out, token);
        }
    }
}

}  // namespace

// ============================================================
// 自由函数
// ============================================================
std::string normalize_tool_arguments(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 16);
    size_t i = 0;
    json_normalize_value(raw, i, out);
    return out;
}

bool parse_qwen3_tool_call_body(const std::string& body, ToolCall& out) {
    // ---- name ----
    const std::string name_key = "\"name\"";
    size_t np = body.find(name_key);
    if (np == std::string::npos) return false;
    size_t colon = body.find(':', np + name_key.size());
    if (colon == std::string::npos) return false;
    size_t q1 = body.find('"', colon + 1);
    if (q1 == std::string::npos) return false;
    // 简单字符串：不处理转义；Qwen3 工具名是英文标识符，不会含 \"
    size_t q2 = body.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    out.name.assign(body, q1 + 1, q2 - q1 - 1);

    // ---- arguments ----
    const std::string args_key = "\"arguments\"";
    size_t ap = body.find(args_key, q2 + 1);
    if (ap == std::string::npos) return false;
    size_t acol = body.find(':', ap + args_key.size());
    if (acol == std::string::npos) return false;

    // 跳过空白后取第一个非空白字符判断类型
    size_t i = acol + 1;
    while (i < body.size() && (body[i] == ' ' || body[i] == '\t'
                            || body[i] == '\n' || body[i] == '\r')) {
        ++i;
    }
    if (i >= body.size()) return false;

    if (body[i] == '{' || body[i] == '[') {
        // 对象 / 数组：花括号或方括号匹配
        char open  = body[i];
        char close = (open == '{') ? '}' : ']';
        int depth = 0;
        bool in_str = false;
        bool escape = false;
        size_t start = i;
        for (; i < body.size(); ++i) {
            char c = body[i];
            if (in_str) {
                if (escape) escape = false;
                else if (c == '\\') escape = true;
                else if (c == '"')  in_str = false;
            } else {
                if (c == '"') in_str = true;
                else if (c == open)  ++depth;
                else if (c == close) {
                    if (--depth == 0) {
                        // 截取原始字面量后经容错规整器修复成合法 JSON
                        std::string raw(body, start, i - start + 1);
                        out.arguments = normalize_tool_arguments(raw);
                        return true;
                    }
                }
            }
        }
        return false;  // 未闭合
    } else if (body[i] == '"') {
        // 字符串字面量：原样保留（含两侧引号）
        size_t start = i;
        bool escape = false;
        for (++i; i < body.size(); ++i) {
            char c = body[i];
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') {
                out.arguments.assign(body, start, i - start + 1);
                return true;
            }
        }
        return false;
    }

    // 其它字面量（数字 / true / false / null）：扫到 , 或 } 为止
    size_t start = i;
    for (; i < body.size() && body[i] != ',' && body[i] != '}'; ++i) {}
    out.arguments.assign(body, start, i - start);
    // trim 末尾空白
    while (!out.arguments.empty()
           && (out.arguments.back() == ' ' || out.arguments.back() == '\n'
               || out.arguments.back() == '\r' || out.arguments.back() == '\t')) {
        out.arguments.pop_back();
    }
    return !out.arguments.empty();
}

// ============================================================
// ToolCallSplitter
// ============================================================
void ToolCallSplitter::feed(const std::string& chunk,
                            std::string& content_delta,
                            std::vector<ToolCall>& completed_calls) {
    content_delta.clear();
    completed_calls.clear();
    if (chunk.empty() && holdback_.empty() && body_.empty()) return;

    // 拼接 holdback + 本次输入；从头扫描状态机
    std::string buf = std::move(holdback_);
    holdback_.clear();
    buf.append(chunk);

    size_t cursor = 0;
    const std::string kBeg = "<tool_call>";
    const std::string kEnd = "</tool_call>";

    while (cursor < buf.size()) {
        if (!in_call_) {
            size_t pos = buf.find(kBeg, cursor);
            if (pos != std::string::npos) {
                // 把起始标记之前的字节作为 content
                content_delta.append(buf, cursor, pos - cursor);
                cursor = pos + kBeg.size();
                // 吃掉起始标记后的一个换行（Qwen3 模板）
                while (cursor < buf.size() && (buf[cursor] == '\n' || buf[cursor] == '\r')) {
                    ++cursor;
                }
                in_call_ = true;
                body_.clear();
                continue;
            }
            // 没找到完整起始标记 —— 检查末尾是否为部分前缀，留 holdback
            size_t hold = compute_holdback_len(buf, cursor, kBeg);
            if (hold > 0) {
                content_delta.append(buf, cursor, buf.size() - cursor - hold);
                holdback_.assign(buf, buf.size() - hold, hold);
            } else {
                content_delta.append(buf, cursor, buf.size() - cursor);
            }
            cursor = buf.size();
        } else {
            size_t pos = buf.find(kEnd, cursor);
            if (pos != std::string::npos) {
                body_.append(buf, cursor, pos - cursor);
                cursor = pos + kEnd.size();
                // 吃掉闭合标记后的换行
                while (cursor < buf.size() && (buf[cursor] == '\n' || buf[cursor] == '\r')) {
                    ++cursor;
                }
                ToolCall tc;
                tc.id = make_id();
                if (parse_qwen3_tool_call_body(body_, tc)) {
                    completed_calls.push_back(std::move(tc));
                }
                body_.clear();
                in_call_ = false;
                continue;
            }
            // 没找到完整闭合标记 —— 末尾可能是部分前缀，留 holdback
            size_t hold = compute_holdback_len(buf, cursor, kEnd);
            if (hold > 0) {
                body_.append(buf, cursor, buf.size() - cursor - hold);
                holdback_.assign(buf, buf.size() - hold, hold);
            } else {
                body_.append(buf, cursor, buf.size() - cursor);
            }
            cursor = buf.size();
        }
    }
}

void ToolCallSplitter::flush(std::string& content_delta,
                             std::vector<ToolCall>& completed_calls) {
    content_delta.clear();
    completed_calls.clear();
    if (!in_call_) {
        content_delta.swap(holdback_);
    } else {
        // 未闭合的 tool_call：当成 content fallback 出来，便于调试
        content_delta = "<tool_call>";
        content_delta += body_;
        content_delta += holdback_;
        body_.clear();
        holdback_.clear();
    }
}

std::string ToolCallSplitter::make_id() {
    std::string id = "call_";
    id += std::to_string(next_id_++);
    return id;
}

size_t ToolCallSplitter::compute_holdback_len(const std::string& buf,
                                              size_t from,
                                              const std::string& marker) {
    size_t avail = buf.size() - from;
    size_t max_hold = std::min(avail, marker.size() - 1);
    size_t hold = 0;
    for (size_t k = 1; k <= max_hold; ++k) {
        if (buf.compare(buf.size() - k, k, marker, 0, k) == 0) {
            hold = k;
        }
    }
    return hold;
}

// ============================================================
// ThinkSplitter
// ============================================================
void ThinkSplitter::feed(const std::string& chunk,
                         std::string& thinking_delta,
                         std::string& content_delta) {
    thinking_delta.clear();
    content_delta.clear();
    if (chunk.empty() && holdback_.empty()) return;

    if (!in_thinking_) {
        content_delta = chunk;
        return;
    }

    std::string buf;
    buf.reserve(holdback_.size() + chunk.size());
    buf.append(holdback_);
    buf.append(chunk);
    holdback_.clear();

    const std::string kEndTag = "</think>";
    size_t pos = buf.find(kEndTag);
    if (pos != std::string::npos) {
        thinking_delta.assign(buf, 0, pos);
        size_t after = pos + kEndTag.size();
        // 吃掉紧随的换行（Qwen 模板里是 "\n\n"）
        while (after < buf.size() && (buf[after] == '\n' || buf[after] == '\r')) {
            ++after;
        }
        content_delta.assign(buf, after, std::string::npos);
        in_thinking_ = false;
        return;
    }

    // 没出现完整标记。检查 buf 末尾是否可能是 "</think>" 的前缀，
    // 是则把它留到下一次 feed 再判断，避免误把 "</thi" 当作 thinking 输出。
    const size_t max_hold = std::min(buf.size(), kEndTag.size() - 1);
    size_t hold_len = 0;
    for (size_t k = 1; k <= max_hold; ++k) {
        if (buf.compare(buf.size() - k, k, kEndTag, 0, k) == 0) {
            hold_len = k;
        }
    }
    if (hold_len > 0) {
        holdback_.assign(buf, buf.size() - hold_len, hold_len);
        thinking_delta.assign(buf, 0, buf.size() - hold_len);
    } else {
        thinking_delta = std::move(buf);
    }
}

void ThinkSplitter::flush(std::string& thinking_delta) {
    thinking_delta = std::move(holdback_);
    holdback_.clear();
}

// ============================================================
// Qwen3PromptBuilder
// ============================================================
std::string Qwen3PromptBuilder::build(const std::vector<Message>& messages,
                                      bool enable_thinking) const {
    std::stringstream ss;

    for (const auto& msg : messages) {
        ss << "<|im_start|>" << role_to_string(msg.role) << "\n";
        ss << msg.content;
        for (const auto& call : msg.tool_calls) {
            ss << "\n<tool_call>\n{\"name\": \"" << call.name
               << "\", \"arguments\": " << call.arguments << "}\n</tool_call>";
        }
        ss << "<|im_end|>\n";
    }

    ss << "<|im_start|>assistant\n";
    if (enable_thinking) {
        ss << "<think>\n";
    } else {
        ss << "<think>\n\n</think>\n\n";
    }
    return ss.str();
}

const char* Qwen3PromptBuilder::role_to_string(Role role) {
    switch (role) {
        case Role::SYSTEM:    return "system";
        case Role::USER:      return "user";
        case Role::ASSISTANT: return "assistant";
        case Role::TOOL:      return "tool";
    }
    return "user";
}

}  // namespace Frostfall
