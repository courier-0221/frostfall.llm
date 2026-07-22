// 流式接口端到端测试（关闭深度思考）
//
// 全部用例统一：stream=true, enable_thinking=false
// 覆盖：
//   1. 简单单轮
//   2. Tool Calling 单轮
//   3. 多轮（含 tool 结果回填）
//
// 公共逻辑（打印 / 帧计数 / CLI 解析 / runner）见 infer_test_util.h。

#include "infer_test_util.h"
#include "inference_engine.h"
#include "llm_types.h"

#include <iostream>
#include <string>
#include <vector>

using namespace Frostfall;
using namespace fftest;

// ============================================================
// Test 1: 简单单轮 — 流式 + no-thinking
// ============================================================
static void test_simple_single_turn(InferenceEngine& engine) {
    std::cout << "\n========================================\n"
              << "[Stream Test 1] 简单单轮 (stream=true, thinking=false)\n"
              << "========================================\n\n";

    LlmRequest req;
    req.msg_id          = "stream-0001";
    req.enable_thinking = false;
    req.stream          = true;
    req.messages = {
        Message{Role::SYSTEM, "你是一个智能助手，请简洁回答用户的问题。", "", {}},
        Message{Role::USER,   "你好，请用一句话介绍一下你自己。",          "", {}},
    };

    StreamCounters cnt;
    std::cout << "[Streaming]\n";
    LlmResponse resp = engine.Infer(req, make_stream_printer(cnt));
    std::cout << "=====================================================================" << std::endl;
    print_response("Stream Test 1 Final", resp);
    print_counters(cnt);
    print_stats(engine);
    std::cout << "=====================================================================" << std::endl;
}

// ============================================================
// Test 2: Tool Calling — 流式 + no-thinking
// ============================================================
static void test_tool_calling(InferenceEngine& engine) {
    std::cout << "\n========================================\n"
              << "[Stream Test 2] Tool Calling (stream=true, thinking=false)\n"
              << "========================================\n\n";

    LlmRequest req;
    req.msg_id          = "stream-0002";
    req.enable_thinking = false;
    req.stream          = true;
    req.messages = {
        Message{Role::SYSTEM, car_system_prompt(),        "", {}},
        Message{Role::USER,   "帮我打开空调，温度调到24度", "", {}},
    };

    StreamCounters cnt;
    std::cout << "[Streaming]\n";
    LlmResponse resp = engine.Infer(req, make_stream_printer(cnt));
    std::cout << "=====================================================================" << std::endl;
    print_response("Stream Test 2 Final", resp);
    print_counters(cnt);
    print_stats(engine);
    std::cout << "=====================================================================" << std::endl;
}

// ============================================================
// Test 3: 多轮（含 tool 结果回填）— 流式 + no-thinking
// ============================================================
static void test_multi_turn(InferenceEngine& engine) {
    std::cout << "\n========================================\n"
              << "[Stream Test 3] 多轮 + Tool 结果回填 (stream=true, thinking=false)\n"
              << "========================================\n\n";

    // ---------- Round 1: 触发 tool call ----------
    LlmRequest round1;
    round1.msg_id          = "stream-0003-1";
    round1.enable_thinking = false;
    round1.stream          = true;
    round1.messages = {
        Message{Role::SYSTEM, car_system_prompt(), "", {}},
        Message{Role::USER,   "帮我打开空调",       "", {}},
    };

    StreamCounters cnt1;
    std::cout << "[Round 1 Streaming]\n";
    LlmResponse resp1 = engine.Infer(round1, make_stream_printer(cnt1));
    std::cout << "=====================================================================" << std::endl;
    print_response("Round 1 Final", resp1);
    print_counters(cnt1);
    std::cout << "=====================================================================" << std::endl;

    // ---------- Round 2: 回填 tool 结果，追加新需求 ----------
    LlmRequest round2;
    round2.msg_id          = "stream-0003-2";
    round2.enable_thinking = false;
    round2.stream          = true;

    Message asst;
    asst.role       = Role::ASSISTANT;
    asst.content    = "";
    asst.tool_calls = {ToolCall{"call_001", "ACControl", "{\"action\":\"open\"}"}};

    Message tool_msg;
    tool_msg.role         = Role::TOOL;
    tool_msg.tool_call_id = "call_001";
    tool_msg.content      = "{\"status\":\"success\",\"message\":\"空调已打开\"}";

    round2.messages = {
        Message{Role::SYSTEM, car_system_prompt(), "", {}},
        Message{Role::USER,   "帮我打开空调", "", {}},
        asst,
        tool_msg,
        Message{Role::USER,   "温度调到24度", "", {}},
    };

    StreamCounters cnt2;
    std::cout << "[Round 2 Streaming]\n";
    LlmResponse resp2 = engine.Infer(round2, make_stream_printer(cnt2));
    std::cout << "=====================================================================" << std::endl;
    print_response("Round 2 Final", resp2);
    print_counters(cnt2);
    print_stats(engine);
    std::cout << "=====================================================================" << std::endl;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    CliOptions opt;
    if (!parse_args(argc, argv, opt) || opt.help) {
        print_usage(argv[0], "All tests use: stream=true, enable_thinking=false");
        return opt.help ? 0 : 1;
    }

    InferenceEngine engine;
    if (!init_engine(engine, opt.config_path)) return 1;

    TestRunner runner(engine, opt.test_num);
    runner.run(1, "simple single-turn",       test_simple_single_turn);
    runner.run(2, "tool calling",             test_tool_calling);
    runner.run(3, "multi-turn + tool result", test_multi_turn);

    return runner.summary("stream, no-thinking");
}
