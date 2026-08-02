#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Frostfall {

// ============================================================
// 角色 & 消息
// ============================================================

enum class Role {
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL,
};

/// assistant 发起的单次工具调用
struct ToolCall {
    // 工具调用唯一标识，例如 "call_0"、"call_001"。
    // 响应侧：由库在解析 <tool_call>…</tool_call> 时自动生成（call_0, call_1, ...）。
    // 业务层须在下一轮请求中原样填到对应工具结果 Message 的 tool_call_id 字段，
    // 让模型知道这条工具结果属于哪次调用。
    std::string id;
    std::string name;       // 工具名
    std::string arguments;  // 参数：JSON 字符串（保持与模型输出一致，由业务层解析）
};

/// 单条 message（贯穿请求与响应）
struct Message {
    Role        role         = Role::USER;
    std::string content;              // 文本内容；assistant 发起 tool_call 时可为空
    // 仅 role=TOOL 时填写，值须与触发此工具调用的 ToolCall::id 完全一致。
    // 作用：把工具执行结果关联回上一轮 ASSISTANT 消息中对应的 tool_calls[*]，
    // 让模型在多轮对话里正确理解"这条结果属于哪次调用"。
    // 配对示例：
    //   Round N 响应: tool_calls[0].id = "call_0"       <- ToolCall::id（库生成）
    //   Round N+1 请求:
    //     [ASSISTANT] tool_calls=[{id="call_0", ...}]   <- 原样回放
    //     [TOOL]      tool_call_id="call_0", content=.. <- 与上面配对
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;  // 仅 role=ASSISTANT 时填
};

// ============================================================
// 采样参数（引擎级配置，非请求级）
//
// 字段与内部 sampler_params 一一对应；引擎在 Init 时读取，整个生命周期固定。
// 采样链：repeat penalty -> top-k -> temperature+softmax -> top-p -> 随机采样。
// 各步可单独关闭；默认全部关闭 => 贪心解码（argmax）。
// ============================================================
struct SamplingParams {
    float    temp           = 0.0f;        // <=0 => 贪心 argmax（默认），忽略其它采样项
    int32_t  top_k          = 0;           // <=0 => 关闭
    float    top_p          = 1.0f;        // >=1 => 关闭
    float    repeat_penalty = 1.0f;        // ==1 => 关闭
    int32_t  repeat_last_n  = 64;          // <0  => 回看整段历史
    uint32_t seed           = 0xFFFFFFFFu; // 哨兵：运行时随机取值
};

// ============================================================
// 推理请求 / 响应
// ============================================================

/// 一次推理请求（单轮 / 多轮共用此结构）
struct LlmRequest {
    std::string          msg_id;                  // 请求唯一标识，原样回传到 LlmResponse
    std::vector<Message> messages;                // 单轮 / 多轮共用，按时间顺序排列
                                                  // 工具定义由业务自行写进某条 system Message.content
    bool                 enable_thinking = false; // 是否启用 <think>（可选，默认关闭）
    bool                 stream          = false; // true=流式回调，false=一次性返回

    int32_t              max_tokens = 0;           // 本次最多生成 token；<=0 用 EngineConfig.max_tokens
    std::vector<std::string> stop;                 // 占位：额外停止串（除 eos 外）；v1.0 暂不实现，仅 eos 生效
};

/// 终止原因（对齐 OpenAI Chat Completions 的 finish_reason）
enum class FinishReason {
    NONE,        // 流式中间帧
    STOP,        // 自然结束 / 命中 stop token
    TOOL_CALLS,  // 末尾输出了工具调用
    LENGTH,      // 触达 max_tokens
    CANCELLED,   // 业务层通过 InferenceEngine::cancel() 主动打断
    ERROR,       // 底层错误
};

/// 一次推理的响应（流式回调与非流式返回值共用此结构）
///
/// 字段语义随 stream 模式不同：
///   非流式 (stream=false):
///       - 仅在 infer() 返回值中出现一次
///       - message.content / thinking 为完整文本
///       - message.tool_calls 为本次推理产出的全部工具调用（可能为空）
///       - is_end == true；finish_reason 为终止原因
///   流式 (stream=true):
///       - 每段 UTF-8 文本回调一次；thinking / content / tool_calls 三类帧互斥：
///           * thinking 帧：thinking 非空，content/tool_calls 空
///           * content 帧：content 非空，thinking/tool_calls 空
///           * tool_call 帧：tool_calls 含一个完整 ToolCall，thinking/content 空
///       - 中间帧 is_end=false、finish_reason=NONE
///       - 最后一帧 is_end=true，finish_reason 设为终止原因，content/thinking/tool_calls 可能为空
///       - infer() 返回值含累积后的完整 message.content / message.tool_calls / thinking
struct LlmResponse {
    std::string  msg_id;                          // 原样回传请求中的 msg_id
    Message      message;                         // role 固定为 ASSISTANT
    std::string  thinking;                        // <think>..</think> 间内容；未启用 thinking 时为空
    bool         is_end = false;                  // 是否为最后一帧
    FinishReason finish_reason = FinishReason::NONE;  // 终止原因（仅最后一帧 / 非流式返回值有效）
};

/// 流式回调
///
/// 设计要点：
///   - response 以非 const 引用传入：库内部填充字段后调用回调，
///     业务层在回调里直接读取，避免一次 LlmResponse 拷贝。
///   - 业务层不应在回调里改写 response，改写也不会影响后续生成。
///   - 调用频率：流式时每段可显示 UTF-8 文本一次 + 最后一次终止帧；
///     非流式时不会被调用（可传空 std::function）。
using StreamCallback = std::function<void(LlmResponse& response)>;

}  // namespace Frostfall
