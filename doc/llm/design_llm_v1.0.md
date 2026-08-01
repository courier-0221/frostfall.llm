# frostfall.llm v1.0 —— 通用推理接口方案设计

> 在 v0.3（自研分词 + 增量 KV cache + 采样策略）的基础上，v1.0 把「命令行 demo」升级为
> 一套**可被业务集成的通用推理接口**：统一的请求/响应协议、流式与非流式合一、
> thinking 段与 tool call 的自动分离，以及无状态多轮对话。
> 目标模型仍只有 Qwen3-0.6B，CPU backend。

---

## 1. 目标与范围

### 1.1 新增能力
- **统一推理接口** `InferenceEngine`：一个 `Infer()` 同时覆盖流式 / 非流式。
- **标准请求响应协议**（`llm_types.h`）：`Message` / `LlmRequest` / `LlmResponse`，对齐
  OpenAI Chat Completions 的语义（role、finish_reason、tool_calls）。
- **Thinking 分离**：把 `<think>…</think>` 思考段从正文里切出来，单独放进 `LlmResponse.thinking`。
- **Tool Call 分离**：把模型输出里的 `<tool_call>…</tool_call>` 解析成结构化 `ToolCall`。
- **无状态多轮**：业务每次传完整 `messages`（含历史与工具结果），引擎内部每次重置 KV cache。
- **配置化**：`EngineConfig` 支持结构体 / JSON 文件 / JSON 字符串三种初始化入口。
- **线程安全 + 可取消**：`Cancel()` 可跨线程打断当前生成。

### 1.2 关键决策（沿用 base_design_v1.0 §9）
- 采样参数是**引擎级固定配置**（`EngineConfig.sampling`），不做请求级覆盖。
- 工具定义不引入结构化字段，由业务自己写进某条 `system` message 的 `content`。
- `stop` 字段先占位，v1.0 仅 `eos` 生效。
- 所有公共符号收敛到 `namespace Frostfall`。

---

## 2. 模块设计

### 2.1 模块划分

```
inference_engine.{h,cpp}   引擎门面：Init / Infer / Cancel / GetLastStats
llm_types.h                协议类型：Role / Message / ToolCall / LlmRequest / LlmResponse / SamplingParams
qwen3_chat.{h,cpp}         Qwen3 ChatML 编解码：PromptBuilder + ThinkSplitter + ToolCallSplitter
```

`InferenceEngine::Infer()` 是主链路，内部编排如下：

```
messages
  └─ Qwen3PromptBuilder.build ──► ChatML prompt
        └─ tokenizer.encode ──► ids
              └─ prefill + 自回归 decode（逐 token）
                    └─ 每个新 token 的字面量文本
                          └─ ThinkSplitter.feed ──► thinking_delta / after_think
                                └─ ToolCallSplitter.feed(after_think) ──► content_delta / ToolCall
                                      └─ 累积到 LlmResponse ＋（流式时）emit 一帧
```

> 关键点：喂给 splitter 的 token 文本必须用 `id_to_piece(id, skip_special=false)`。
> 因为 `<think>` / `</think>` / `<tool_call>` 都是 `USER_DEFINED`(type 4) token，
> `skip_special=true` 会把它们吞掉，splitter 就看不到标记字面量了。
> `eos`（`<|im_end|>` / `<|endoftext|>`）在解码循环里已提前 `break`，不会喂进 splitter。

### 2.2 ThinkSplitter —— 思考段与正文的分离

**职责**：把生成的文本流拆成 `thinking`（`<think>…</think>` 之间）和 `content`（正文）。

**初始状态**由 `enable_thinking` 决定：
- `enable_thinking=true`：prompt 末尾是 `<think>\n`，模型直接开始输出思考内容，
  所以 splitter 初始即处于 `in_thinking=true`，直到看到 `</think>` 才切到正文。
- `enable_thinking=false`：prompt 已写死 `<think>\n\n</think>\n\n`，模型不产思考，
  splitter 初始 `in_thinking=false`，所有输出都是 content。

**状态机**（单个结束标记 `</think>`）：

```
                       看到 "</think>"
  ┌──────────────┐  ─────────────────►  ┌──────────────┐
  │ in_thinking  │                      │  content     │
  │ 输出 thinking │  ◄── 一去不回 ──     │  输出 content │
  └──────────────┘                      └──────────────┘
```

**跨分片切碎处理（holdback）**：一次 token 解码可能只吐出 `</thi`，标记被 UTF-8/分词切成两段。
splitter 检查 buffer 末尾是否是 `</think>` 的**前缀**，若是就把这几个字节缓存进 `holdback_`，
留到下一次 `feed` 再拼起来判断，避免把 `</thi` 误当作 thinking 正文输出。
推理循环结束时 `flush()` 把残留 holdback 当作 thinking 吐出。

### 2.3 ToolCallSplitter —— 工具调用的分离与结构化

**职责**：从「已剔除 thinking 的正文流」里，把 `<tool_call>…</tool_call>` 抠出来解析成 `ToolCall`，
其余字节作为普通 `content` 下发。业务层永远不会看到半截 JSON。

**状态机**（成对标记 `<tool_call>` / `</tool_call>`）：

```
   看到 "<tool_call>"                      看到 "</tool_call>"
        （之前字节→content）                    （body→解析成 ToolCall）
  ┌────────────┐  ────────────►  ┌────────────┐  ────────────►  回到 NORMAL
  │  NORMAL     │                 │  IN_CALL    │                 分配 id=call_<n>
  │ 输出 content│  ◄────────────  │ 累积 body   │
  └────────────┘                 └────────────┘
```

**处理细节**：
- NORMAL 态：`<tool_call>` 之前的字节作为 `content_delta`；吃掉起始标记后紧跟的换行（Qwen3 模板）。
- IN_CALL 态：累积 body 直到 `</tool_call>`，把整段交给 `parse_qwen3_tool_call_body`
  填出 `ToolCall.name / arguments`；`arguments` 经 `normalize_tool_arguments` 容错规整成合法 JSON
  （例如 `{action=open}` → `{"action":"open"}`）。
- **id 由库自动生成**：`call_0`、`call_1`……业务下一轮须把它原样回填到工具结果 message 的 `tool_call_id`。
- 跨分片同样用 `holdback_`：末尾若是 `<tool_cal` 这类部分标记，缓存等下次拼接。
- `flush()`：残留正文直接吐出；未闭合的 `<tool_call>` 作为 content fallback 出来，便于调试。

### 2.4 帧互斥与终止帧（流式协议）

`Infer(stream=true)` 时，每帧 `thinking` / `content` / `tool_call` **三类互斥**：
- thinking 帧：`thinking` 非空，`content` / `tool_calls` 空；
- content 帧：`content` 非空，其余空；
- tool_call 帧：`tool_calls` 含一个完整 `ToolCall`，其余空。

中间帧 `is_end=false`、`finish_reason=NONE`；最后单独发一个 `is_end=true` 的终止帧，
`finish_reason` 为最终原因。`Infer()` 的返回值始终是累积后的完整 `LlmResponse`。

### 2.5 FinishReason 判定

| 场景 | finish_reason |
| --- | --- |
| 遇到 eos 自然结束，且**没有** tool_call | `STOP` |
| 遇到 eos 自然结束，且本轮产出过 tool_call | `TOOL_CALLS` |
| 触达 `max_tokens` 或 `n_ctx` 上限 | `LENGTH` |
| 生成过程中被 `Cancel()` 打断 | `CANCELLED` |
| 底层错误（未初始化、编码失败等） | `ERROR` |

---

## 3. Message 的构造与使用（参考 examples/llm/api_test）

`Message` 结构（`llm_types.h`）：

```cpp
struct Message {
    Role                  role = Role::USER;   // SYSTEM / USER / ASSISTANT / TOOL
    std::string           content;             // 文本；assistant 发起 tool_call 时可为空
    std::string           tool_call_id;        // 仅 role=TOOL，配对上一轮 ToolCall::id
    std::vector<ToolCall> tool_calls;          // 仅 role=ASSISTANT
};
```

### 3.1 简单单轮

`system` 给出人设，`user` 给出问题，直接 `Infer`：

```cpp
LlmRequest req;
req.msg_id          = "blocking-0001";
req.enable_thinking = false;
req.stream          = false;
req.messages = {
    Message{Role::SYSTEM, "你是一个智能助手，请简洁回答用户的问题。", "", {}},
    Message{Role::USER,   "你好，请用一句话介绍一下你自己。",          "", {}},
};

LlmResponse resp = engine.Infer(req);   // resp.message.content 为完整回答
```

### 3.2 工具调用（单轮）

工具定义写进 `system` message 的 `content`（Qwen3 官方 `<tools>` 模板，见
`car_system_prompt()`）。命中工具时，`resp.message.tool_calls` 里给出结构化调用：

```cpp
LlmRequest req;
req.msg_id   = "blocking-0002";
req.messages = {
    Message{Role::SYSTEM, car_system_prompt(),        "", {}},  // 工具定义在 system.content
    Message{Role::USER,   "帮我打开空调，温度调到24度", "", {}},
};

LlmResponse resp = engine.Infer(req);
// resp.finish_reason == TOOL_CALLS
// resp.message.tool_calls[0] = { id="call_0", name="ACControl", arguments={"action":"open","temperature":24} }
```

### 3.3 多轮 + 工具结果回填

业务执行完工具后，把「assistant 的 tool_calls」与「TOOL 结果」按顺序拼回 `messages`
再发一轮。`TOOL` message 的 `tool_call_id` 必须与上一轮 `ToolCall::id` 一致：

```cpp
// 上一轮 assistant 发起的调用（原样回放）
Message asst;
asst.role       = Role::ASSISTANT;
asst.content    = "";
asst.tool_calls = { ToolCall{"call_001", "ACControl", "{\"action\":\"open\"}"} };

// 工具执行结果，tool_call_id 与上面配对
Message tool_msg;
tool_msg.role         = Role::TOOL;
tool_msg.tool_call_id = "call_001";
tool_msg.content      = "{\"status\":\"success\",\"message\":\"空调已打开\"}";

LlmRequest round2;
round2.msg_id   = "blocking-0003-2";
round2.messages = {
    Message{Role::SYSTEM, car_system_prompt(), "", {}},
    Message{Role::USER,   "帮我打开空调",       "", {}},
    asst,
    tool_msg,
    Message{Role::USER,   "温度调到24度",       "", {}},
};

LlmResponse resp2 = engine.Infer(round2);
```

> 无状态：引擎不跨调用保留会话，多轮全靠业务每次传入完整 `messages`。

### 3.4 流式用法

`stream=true` 并传入回调，每段可显示文本回调一帧，最后一帧 `is_end=true`：

```cpp
LlmRequest req;
req.stream = true;
req.messages = { /* 同上 */ };

engine.Infer(req, [](LlmResponse& chunk) {
    if (!chunk.thinking.empty())            /* 思考帧 */;
    else if (!chunk.message.content.empty()) /* 正文帧 */;
    else if (!chunk.message.tool_calls.empty()) /* 工具调用帧 */;
    if (chunk.is_end) /* 终止帧：读 chunk.finish_reason */;
});
```

---

## 4. 端到端调用示例

示例位于 `examples/llm/api_test/`，按 thinking 开/关 × 流式/非流式分组，公共逻辑在
`infer_test_util.h`（打印、统计、CLI 解析、流式帧互斥校验、runner）。

| 文件 | stream | enable_thinking | 覆盖 |
| --- | --- | --- | --- |
| `infer_nothink_blocking.cpp` | false | false | 简单单轮 / 工具调用 / 多轮含 tool 回填 |
| `infer_nothink_stream.cpp`   | true  | false | 同上，额外校验帧互斥、终止帧唯一、流式累积 == 返回值 |

配置走 JSON（`infer_nothink.json`：`model_path` / `n_ctx` / `n_threads` / `max_tokens` / `sampling`，
`temp=0` 贪心 + `seed=42` 保证可复现），运行须在 repo 根目录（`model_path` 为相对路径）。
