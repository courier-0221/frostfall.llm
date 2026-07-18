#pragma once

// frostfall v0.2 —— 小工具集合：计时、chat 模板、字节数格式化。

#include <chrono>
#include <cstdint>
#include <string>

namespace ff {

// 简单的墙钟计时器：construct 时开始，elapsed_ms() 返回距开始的毫秒数。
struct timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void   reset()      { t0 = std::chrono::steady_clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

// 把字节数格式化成人类可读的 MB 字符串（保留两位小数）。
std::string format_mb(size_t bytes);

// 套用 Qwen3 的 chat 模板（单轮 user 消息）。等价于 transformers 的
//   apply_chat_template([{"role":"user","content":user}], add_generation_prompt=True, enable_thinking=...)
// enable_thinking=false 时会附带空的 <think>\n\n</think>\n\n 块（与 Qwen3 默认模板一致）。
std::string apply_chat_template(const std::string & user, bool enable_thinking);

} // namespace ff
