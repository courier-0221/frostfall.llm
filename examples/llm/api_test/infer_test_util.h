// 推理端到端测试公共工具头
//
// 供 infer_nothink_stream.cpp / infer_nothink_blocking.cpp 等测试共用：
//   - FinishReason 转字符串
//   - 响应 / 统计打印
//   - 车载工具调用 system prompt 常量
//   - 流式帧计数器 + 回调工厂（含帧互斥校验）
//   - 命令行解析（--config / --test / --help）
//   - 测试 runner 骨架
//
// 全部函数 inline，避免多个测试可执行文件各自包含时的重复定义问题。
#pragma once

#include "inference_engine.h"
#include "llm_types.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

namespace fftest {

// ============================================================
// 车载工具调用 system prompt（工具调用类用例共用）
//
// 采用 Qwen3 官方工具调用模板：<tools> 内给出函数签名，并要求模型把每次调用
// 用 <tool_call>{"name":..,"arguments":..}</tool_call> 包裹。这与框架内
// ToolCallSplitter / Qwen3PromptBuilder 期望的格式一致，贪心解码下可稳定解析。
// （若改成“仅输出工具JSON”会得到 markdown ```json 代码块，框架无法识别为 tool_call。）
// ============================================================
inline const std::string& car_system_prompt() {
    static const std::string kPrompt =
        "你是车载语音助手。请根据用户请求判断是否调用工具；范围外的请求直接用自然语言拒绝。\n\n"
        "# Tools\n\n"
        "你可以调用一个或多个函数来协助完成用户请求。\n\n"
        "以下是可用函数的签名，包含在 <tools></tools> 标签中：\n"
        "<tools>\n"
        "{\"type\": \"function\", \"function\": {\"name\": \"ACControl\", "
        "\"description\": \"空调控制\", \"parameters\": {\"type\": \"object\", "
        "\"properties\": {\"action\": {\"type\": \"string\", \"enum\": [\"open\", \"close\", \"adjust\"]}, "
        "\"temperature\": {\"type\": \"integer\"}}, \"required\": [\"action\"]}}}\n"
        "{\"type\": \"function\", \"function\": {\"name\": \"Navigation\", "
        "\"description\": \"导航控制\", \"parameters\": {\"type\": \"object\", "
        "\"properties\": {\"destination\": {\"type\": \"string\"}}, \"required\": [\"destination\"]}}}\n"
        "</tools>\n\n"
        "对于每次函数调用，返回一个包含函数名和参数的 JSON 对象，并用 <tool_call></tool_call> 标签包裹：\n"
        "<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>";
    return kPrompt;
}

// ============================================================
// FinishReason -> 字符串
// ============================================================
inline const char* finish_reason_str(Frostfall::FinishReason r) {
    using Frostfall::FinishReason;
    switch (r) {
        case FinishReason::NONE:       return "NONE";
        case FinishReason::STOP:       return "STOP";
        case FinishReason::TOOL_CALLS: return "TOOL_CALLS";
        case FinishReason::LENGTH:     return "LENGTH";
        case FinishReason::ERROR:      return "ERROR";
        case FinishReason::CANCELLED:  return "CANCELLED";
    }
    return "?";
}

// ============================================================
// 打印一条最终响应
// ============================================================
inline void print_response(const std::string& tag, const Frostfall::LlmResponse& resp) {
    std::cout << "[" << tag << " msg_id=" << resp.msg_id
              << " finish_reason=" << finish_reason_str(resp.finish_reason)
              << " is_end=" << (resp.is_end ? "true" : "false") << "]\n";
    if (!resp.thinking.empty()) {
        std::cout << "  [thinking] " << resp.thinking << std::endl;
    }
    std::cout << "  [content] " << resp.message.content << std::endl;
    for (size_t i = 0; i < resp.message.tool_calls.size(); ++i) {
        const auto& tc = resp.message.tool_calls[i];
        std::cout << "  [tool_call #" << i << "] id=" << tc.id
                  << " name=" << tc.name
                  << " arguments=" << tc.arguments << std::endl;
    }
}

// ============================================================
// 打印最近一次推理的性能统计
// ============================================================
inline void print_stats(Frostfall::InferenceEngine& engine) {
    auto s = engine.GetLastStats();
    std::cout << "  [stats] prompt=" << s.prompt_tokens_per_sec << " tok/s, "
              << "gen=" << s.gen_tokens_per_sec << " tok/s, "
              << "gen_tokens=" << s.gen_token_count
              << " prompt_tokens=" << s.prompt_token_count
              << " ttft=" << s.ttft_ms << " ms\n";
}

// ============================================================
// 流式帧计数器 + 回调工厂
// ============================================================
struct StreamCounters {
    int content_frames   = 0;
    int thinking_frames  = 0;
    int tool_call_frames = 0;
    int end_frames       = 0;
    bool frame_exclusive = true;  // 每帧 thinking/content/tool_call 三选一互斥
};

/// 构造一个把流式分片打印到 stdout 并统计帧数的回调。
/// 中间帧要求 thinking / content / tool_calls 三类互斥，违反则置 frame_exclusive=false。
inline Frostfall::StreamCallback make_stream_printer(StreamCounters& cnt) {
    return [&cnt](Frostfall::LlmResponse& chunk) {
        const bool has_think = !chunk.thinking.empty();
        const bool has_text  = !chunk.message.content.empty();
        const bool has_tool  = !chunk.message.tool_calls.empty();

        // 三类内容帧互斥校验（仅对含内容的帧检查；纯终止帧可全空）
        int kinds = (has_think ? 1 : 0) + (has_text ? 1 : 0) + (has_tool ? 1 : 0);
        if (kinds > 1) cnt.frame_exclusive = false;

        if (has_think) {
            std::cout << "[streamcb think] " << chunk.thinking << std::endl;
            ++cnt.thinking_frames;
        }
        if (has_text) {
            std::cout << "[streamcb content] " << chunk.message.content << std::endl;
            ++cnt.content_frames;
        }
        if (has_tool) {
            ++cnt.tool_call_frames;
            for (const auto& tc : chunk.message.tool_calls) {
                std::cout << "[streamcb tool_call] id=" << tc.id
                          << " name=" << tc.name
                          << " args=" << tc.arguments << std::endl;
            }
        }
        if (chunk.is_end) ++cnt.end_frames;
    };
}

// ============================================================
// 打印一次流式推理的帧计数汇总
// ============================================================
inline void print_counters(const StreamCounters& cnt) {
    std::cout << "  [frames] content=" << cnt.content_frames
              << " thinking=" << cnt.thinking_frames
              << " tool_call=" << cnt.tool_call_frames
              << " end=" << cnt.end_frames
              << " exclusive=" << (cnt.frame_exclusive ? "true" : "false") << "\n";
}

// ============================================================
// 命令行解析
// ============================================================
struct CliOptions {
    std::string config_path;  // JSON 配置文件路径（必填）
    int         test_num = 0; // 0 = 运行全部；否则只运行指定编号
    bool        help     = false;
};

inline void print_usage(const char* prog, const char* mode_desc) {
    std::cout << "Usage: " << prog << " --config <json_path> [--test <num>]\n"
              << mode_desc << "\n"
              << "  --config <json_path>  引擎配置（JSON）\n"
              << "  --test   <num>        只运行第 num 个用例（默认 0=全部）\n";
}

/// 解析 argv。返回 false 表示参数非法（缺 --config）。
inline bool parse_args(int argc, char* argv[], CliOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--config" && i + 1 < argc) {
            opt.config_path = argv[++i];
        } else if (arg == "--test" && i + 1 < argc) {
            opt.test_num = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            opt.help = true;
            return true;
        }
    }
    return !opt.config_path.empty();
}

// ============================================================
// 测试 runner 骨架
// ============================================================
struct TestRunner {
    Frostfall::InferenceEngine& engine;
    int selected;      // 0 = 全部
    int total_count = 0;

    TestRunner(Frostfall::InferenceEngine& e, int selected_test)
        : engine(e), selected(selected_test) {}

    /// fn 签名：void(InferenceEngine&)；用例只跑一遍并打印相关状态，不做通过/失败判定
    template <typename Fn>
    void run(int num, const char* label, Fn&& fn) {
        if (selected != 0 && selected != num) return;
        ++total_count;
        fn(engine);
        std::cout << "\n>>> Test " << num << " [" << label << "] done.\n";
    }

    /// 打印汇总，固定返回退出码 0
    int summary(const char* group) const {
        std::cout << "\n========================================\n"
                  << "Ran " << total_count << " tests (" << group << ").\n"
                  << "========================================\n";
        return 0;
    }
};

/// 统一初始化引擎：读取配置文件并加载模型。失败返回 false。
inline bool init_engine(Frostfall::InferenceEngine& engine, const std::string& config_path) {
    std::cout << "Initializing InferenceEngine with: " << config_path << "\n";
    if (!engine.InitFromJsonFile(config_path)) {
        std::cerr << "Failed to initialize InferenceEngine!\n";
        return false;
    }
    std::cout << "Engine initialized. max_tokens=" << engine.MaxTokens() << "\n";
    return true;
}

}  // namespace fftest
