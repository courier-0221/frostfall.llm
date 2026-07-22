#pragma once

#include "llm_types.h"

#include <cstddef>
#include <string>
#include <vector>

namespace Frostfall {

// ============================================================
// Qwen3 聊天协议编解码
//
// 本模块负责 Qwen3 ChatML 的「拼 prompt」与「解析模型输出」：
//   - Qwen3PromptBuilder：messages -> ChatML prompt
//   - ThinkSplitter：从生成流中分离 <think> 思考段与正文
//   - ToolCallSplitter：从正文流中切出 <tool_call> 工具调用
//   - 若干 tool_call 文本解析 / JSON 容错规整的自由函数
// ============================================================

// 把（可能是降级格式的）arguments 字面量规整成合法 JSON：
//   {action=关闭, device=车窗}  ->  {"action":"关闭","device":"车窗"}
// 本就合法的 JSON 经过规整后仍是等价的紧凑合法 JSON。
std::string normalize_tool_arguments(const std::string& raw);

// 解析 Qwen3 <tool_call> 内部的 {"name":..,"arguments":..} body，
// 填入 ToolCall::name / arguments（arguments 经容错规整成合法 JSON）。
bool parse_qwen3_tool_call_body(const std::string& body, ToolCall& out);

// ============================================================
// ToolCallSplitter — 从（已剔除 thinking 的）文本流中切出
// 普通 content 与已完成的 ToolCall。
//
// 规则：
//   * NORMAL 状态：把确定属于 content 的字节下发；若 buffer 末尾形如
//     "<tool_cal" 的部分起始标记，则 holdback 留到下次 feed 再判断。
//   * 看到完整 "<tool_call>"：之前的字节作为 content，切到 IN_CALL。
//   * IN_CALL 状态：累积 body 直到 "</tool_call>"，整段交给
//     parse_qwen3_tool_call_body，分配 id="call_<n>"，作为一帧 tool_call 给出。
// 业务层不会看到半截 JSON。
// ============================================================
class ToolCallSplitter {
public:
    ToolCallSplitter() = default;

    /// 输入一段文本，输出：
    ///   - content_delta:   本次新增的、确定属于 content 的字节
    ///   - completed_calls: 本次 feed 完成解析的 ToolCall 列表
    void feed(const std::string& chunk,
              std::string& content_delta,
              std::vector<ToolCall>& completed_calls);

    /// 推理循环结束时调用：把残留作为 content 吐出（不强行闭合未完成的 tool_call）。
    void flush(std::string& content_delta,
               std::vector<ToolCall>& completed_calls);

    bool in_tool_call() const { return in_call_; }
    int  emitted_count() const { return next_id_; }

private:
    std::string make_id();

    /// 计算 buf[from..end] 末尾与 marker 的最长前缀重叠长度（用于跨段切碎）
    static size_t compute_holdback_len(const std::string& buf,
                                       size_t from,
                                       const std::string& marker);

    bool        in_call_ = false;
    std::string body_;       // IN_CALL 时累积的 tool_call 内部 JSON
    std::string holdback_;   // 跨段切碎缓冲
    int         next_id_ = 0;
};

// ============================================================
// ThinkSplitter — 从生成的文本流中分离 thinking / content。
//
// enable_thinking=true 时，模型从 "<think>" 段之后开始生成，直到输出
// "</think>" 切出正文；=false 时 prompt 已写死 "<think>\n\n</think>\n\n"，
// 模型直接生成正文。用 holdback 缓冲处理 "</think>" 被 UTF-8 分段切碎的情况。
// ============================================================
class ThinkSplitter {
public:
    explicit ThinkSplitter(bool initial_in_thinking)
        : in_thinking_(initial_in_thinking) {}

    /// 输入一段已经过 UTF-8 整理的文本，拆成本次的 thinking_delta / content_delta。
    void feed(const std::string& chunk,
              std::string& thinking_delta,
              std::string& content_delta);

    /// 推理循环结束时调用，把残留的 holdback 当作 thinking 输出。
    void flush(std::string& thinking_delta);

    bool in_thinking() const { return in_thinking_; }

private:
    bool        in_thinking_;
    std::string holdback_;
};

// ============================================================
// Qwen3PromptBuilder — Qwen3 ChatML 拼接器
//
// 工具定义直接写在 system message 的 content 中；assistant 发起的工具调用
// 渲染为 <tool_call> 块，与 Qwen3 训练时的 prompt 格式一致。
// ============================================================
class Qwen3PromptBuilder {
public:
    /// 将 messages 拼接为 Qwen3 ChatML prompt。
    /// @param enable_thinking true  → assistant 段以 "<think>\n" 开头，模型继续生成思考
    ///                        false → 追加 "<think>\n\n</think>\n\n" 跳过思考
    std::string build(const std::vector<Message>& messages, bool enable_thinking) const;

private:
    static const char* role_to_string(Role role);
};

}  // namespace Frostfall
