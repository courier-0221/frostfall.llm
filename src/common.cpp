#include "common.h"

#include <cstdio>

namespace ff {

std::string format_mb(size_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f MB", bytes / 1024.0 / 1024.0);
    return buf;
}

std::string apply_chat_template(const std::string & user, bool enable_thinking) {
    std::string s;
    s += "<|im_start|>user\n";
    s += user;
    s += "<|im_end|>\n";
    s += "<|im_start|>assistant\n";
    if (!enable_thinking) {
        // Qwen3 模板在关闭思考模式时会补一个空的 think 块。
        s += "<think>\n\n</think>\n\n";
    }
    return s;
}

} // namespace ff
