# frostfall.llm

> 基于 **ggml v0.15.3** 的极简教学版大模型推理框架，唯一目标模型：**Qwen3-0.6B**。
>
> 设计目标：用尽量少、尽量清晰的 C++ 代码，把"加载模型 → 分词 → 构图 → 前向 → 增量解码"的完整链路跑通。

---

## 当前版本：v1.0

v1.0 在 v0.3（自研分词 + 增量 KV cache + 采样策略）的基础上，把"命令行 demo"升级为
**可被业务集成的通用推理接口**：命令行程序 `frostfall` 已下线，仓库产出改为共享库
`libfrostfall.so` + 一组调用示例（`examples/llm/api_test`）。

- **统一推理接口** `Frostfall::InferenceEngine`：一个 `Infer()` 同时覆盖流式 / 非流式，签名对齐
  OpenAI Chat Completions 的语义（`role`、`finish_reason`、`tool_calls`）。
- **Thinking 分离**：`<think>…</think>` 思考段自动从正文中切出，单独放进 `LlmResponse.thinking`。
- **Tool Call 分离**：模型输出里的 `<tool_call>…</tool_call>` 自动解析成结构化 `ToolCall`。
- **无状态多轮对话**：业务每次传入完整 `messages`（含历史与工具结果），引擎内部每次 `Infer` 自动重置 KV cache。
- **配置化初始化**：`EngineConfig` 支持结构体 / JSON 文件 / JSON 字符串三种初始化入口。
- **线程安全 + 可取消**：`Cancel()` 可在任意线程调用，打断正在进行的生成。

v0.1~v0.3 的自研 Tokenizer、增量 KV cache、采样策略（temperature / top-k / top-p / 重复惩罚 / seed）
均在库内部保留，只是不再通过命令行参数暴露，而是通过 `EngineConfig.sampling`（引擎级固定配置）传入。


---

## 目录结构

```
frostfall.llm/
├── src/
│   ├── inference_engine.{h,cpp}  # 引擎门面：Init / Infer / Cancel / GetLastStats
│   ├── llm_types.h               # 协议类型：Role/Message/ToolCall/LlmRequest/LlmResponse/SamplingParams
│   ├── qwen3_chat.{h,cpp}        # Qwen3 ChatML 编解码：PromptBuilder + ThinkSplitter + ToolCallSplitter
│   ├── model.{h,cpp}             # GGUF 加载，qwen3_model 权重结构体
│   ├── graph.{h,cpp}             # qwen3_build_graph()，搭出 Qwen3 前向计算图
│   ├── kv_cache.{h,cpp}          # 增量 KV cache
│   ├── tokenizer.{h,cpp}         # 自研 byte-level BPE 分词器（从 GGUF 元数据构建）
│   ├── sampler.{h,cpp}           # 采样策略（greedy / temperature / top-k / top-p / 重复惩罚）
│   ├── common.{h,cpp}            # 计时器、字节数格式化等工具
│   └── log.h                     # 日志宏
├── examples/llm/api_test/          # 链接 frostfall 库的端到端调用示例（同时也是 CMake target）
│   ├── infer_nothink_blocking.cpp  # 非流式 + 关闭思考：单轮/工具调用/多轮
│   ├── infer_nothink_stream.cpp    # 流式 + 关闭思考：单轮/工具调用/多轮
│   ├── infer_yesthink_blocking.cpp # 非流式 + 开启思考
│   ├── infer_yesthink_stream.cpp   # 流式 + 开启思考
│   ├── infer_test_util.h           # 公共工具：打印/统计/CLI 解析/runner
│   └── llm_infer_conf.json         # 示例 EngineConfig
├── tools/                # HF -> GGUF 转换脚本
├── third_party/ggml/     # ggml v0.15.3（内置源码，随仓库跟踪）
├── doc/llm/design_llm.md         # 基础推理链路设计文档（v0.1~v0.3）
├── doc/llm/design_llm_v1.0.md    # 通用推理接口设计文档（v1.0，InferenceEngine）
├── doc/llm/code_analysis_llm-v0.1.md  # v0.1 逐接口代码走读
├── doc/llm/code_analysis_llm-v0.2.md  # v0.2 逐接口代码走读
├── doc/llm/code_analysis_llm-v0.3.md  # v0.3 采样策略逐接口走读
└── CMakeLists.txt
```

---

## 环境依赖

构建库本身与全部示例可执行文件只需要 C++ 工具链：

| 依赖 | 说明 |
| --- | --- |
| CMake ≥ 3.14 | 构建系统 |
| C++17 编译器 | GCC 9+ 或 Clang 10+ |

以下依赖仅在做 HF → GGUF 模型转换时才需要：

| 依赖 | 说明 |
| --- | --- |
| Python 3.8+ / `transformers` | 仅用于 `tools/convert_hf_to_gguf.py` 转换模型 |
| Qwen3-0.6B HF 模型 | 转换为 GGUF 后推理 |

---

## 构建

```bash
git clone <repo>
cd frostfall.llm

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 产物
./build/libfrostfall.so                             # 共享库
./build/examples/llm/api_test/infer_nothink_blocking    # 调用示例可执行文件（共 4 个）
```

---

## 准备模型（HF → GGUF）

推理需要 F16 GGUF 格式，用 llama.cpp 的转换脚本生成（GGUF 中会内置 tokenizer 元数据，
库内部推理时无需再依赖 HF 模型目录）：

```bash
python tools/convert_hf_to_gguf.py \
    /path/to/Qwen3-0.6B \
    --outfile models/qwen3-0.6b-f16.gguf \
    --outtype f16
```

---

## 使用（库 API）

对外头文件只需要 `#include "inference_engine.h"` 和 `#include "llm_types.h"`，
链接 `frostfall` 库即可。全部公共符号位于 `namespace Frostfall`。

### 1. 初始化 `InferenceEngine`

`EngineConfig` 可直接构造结构体，也可以从 JSON 文件 / JSON 字符串初始化：

```json
{
  "model_path": "models/qwen3-0.6b-f16.gguf",
  "n_ctx": 4096,
  "n_threads": 4,
  "max_tokens": 512,
  "sampling": {
    "temp": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "seed": 42
  }
}
```

```cpp
#include "inference_engine.h"
#include "llm_types.h"

using namespace Frostfall;

InferenceEngine engine;
if (!engine.InitFromJsonFile("examples/llm/api_test/llm_infer_conf.json")) {
    // 加载模型/初始化失败
}
```

`sampling` 是**引擎级固定配置**（`temp<=0` 即贪心 argmax，默认关闭其余采样项），不支持按请求覆盖。

### 2. 非流式（blocking）推理

```cpp
LlmRequest req;
req.msg_id          = "req-0001";
req.enable_thinking = false;   // 是否开启 <think> 思考段
req.stream          = false;   // 非流式：一次性拿完整结果
req.messages = {
    Message{Role::SYSTEM, "你是一个智能助手，请简洁回答用户的问题。", "", {}},
    Message{Role::USER,   "你好，请用一句话介绍一下你自己。",          "", {}},
};

LlmResponse resp = engine.Infer(req);
std::cout << resp.message.content << std::endl;   // 完整正文
std::cout << resp.thinking << std::endl;           // 完整思考段（未开启则为空）
```

### 3. 流式推理

`stream=true` 时通过回调逐帧接收；`thinking` / `content` / `tool_calls` 三类帧互斥：

```cpp
LlmRequest req;
req.msg_id          = "req-0002";
req.enable_thinking = false;
req.stream          = true;
req.messages = { /* 同上 */ };

LlmResponse final_resp = engine.Infer(req, [](LlmResponse& chunk) {
    if (!chunk.message.content.empty()) std::cout << chunk.message.content;
    if (chunk.is_end) std::cout << "\n[finish_reason=" << (int)chunk.finish_reason << "]\n";
});
```

### 4. Tool Calling

工具定义写进某条 `system` message 的 `content`（Qwen3 官方 `<tools>` 模板），
模型输出的 `<tool_call>` 会被自动解析进 `resp.message.tool_calls`；
下一轮把该次调用的 `ToolCall::id` 原样填进对应工具结果 message 的 `tool_call_id`：

```cpp
// Round 1：触发 tool call
LlmResponse resp1 = engine.Infer(round1);
// resp1.message.tool_calls[0] = { id="call_0", name="ACControl", arguments="{\"action\":\"open\"}" }

// Round 2：回填工具结果
Message asst;
asst.role       = Role::ASSISTANT;
asst.tool_calls = { resp1.message.tool_calls[0] };

Message tool_msg;
tool_msg.role         = Role::TOOL;
tool_msg.tool_call_id = resp1.message.tool_calls[0].id;   // 与上面配对
tool_msg.content      = "{\"status\":\"success\"}";

round2.messages = { system_msg, user_msg, asst, tool_msg, next_user_msg };
LlmResponse resp2 = engine.Infer(round2);
```

### 5. 取消 / 统计

```cpp
engine.Cancel();                       // 任意线程调用，打断正在进行的 Infer()
InferenceStats s = engine.GetLastStats();  // prompt/gen 吞吐、TTFT 等（调试用，非协议字段）
```

更完整的用例（单轮 / 工具调用 / 多轮，流式与非流式、开启与关闭思考模式的全组合）见
`examples/llm/api_test/` 下的四个 `.cpp` 文件。

---

## 运行调用示例（examples/llm/api_test）

每个 `.cpp` 编译成独立可执行文件，统一读取 JSON 配置文件、可选只跑某一个用例：

```bash
./build/examples/llm/api_test/infer_nothink_blocking --config examples/llm/api_test/llm_infer_conf.json
./build/examples/llm/api_test/infer_nothink_stream   --config examples/llm/api_test/llm_infer_conf.json
./build/examples/llm/api_test/infer_yesthink_blocking --config examples/llm/api_test/llm_infer_conf.json
./build/examples/llm/api_test/infer_yesthink_stream   --config examples/llm/api_test/llm_infer_conf.json

# 只跑指定编号的用例（1=简单单轮 2=Tool Calling 3=多轮）
./build/examples/llm/api_test/infer_nothink_blocking --config examples/llm/api_test/llm_infer_conf.json --test 2
```

---

## API 参考（核心类型速查）

### `EngineConfig`

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `model_path` | GGUF 路径（必填） | — |
| `n_ctx` | KV cache 上限（会被 `n_ctx_train` 截断） | `4096` |
| `n_threads` | CPU 计算线程数 | `4` |
| `max_tokens` | 请求未指定 `max_tokens` 时的默认上限 | `512` |
| `sampling` | 引擎级固定采样参数（`SamplingParams`） | 见下 |

### `SamplingParams`（采样链：重复惩罚 → top-k → temperature+softmax → top-p → 随机采样）

| 字段 | 含义 | 默认 | 关闭取值 |
| --- | --- | --- | --- |
| `temp` | 采样温度；`<=0` 退化为贪心 argmax | `0` | `<=0` |
| `top_k` | 只在 logit 最高的 N 个候选里采样 | `0` | `<=0` |
| `top_p` | nucleus 采样：保留累计概率达到 P 的最小候选集 | `1.0` | `>=1.0` |
| `repeat_penalty` | 重复惩罚系数，`>1` 抑制近期出现过的 token | `1.0` | `1.0` |
| `repeat_last_n` | 重复惩罚回看窗口；`<0` 表示整段历史 | `64` | — |
| `seed` | 随机种子；哨兵值 `0xFFFFFFFF` 表示运行时随机取值 | 随机 | — |

### `LlmRequest`

| 字段 | 含义 |
| --- | --- |
| `msg_id` | 请求唯一标识，原样回传到 `LlmResponse` |
| `messages` | 单轮/多轮共用，按时间顺序排列；工具定义写进某条 `system` message |
| `enable_thinking` | 是否启用 `<think>` 思考段（默认关闭） |
| `stream` | `true`=流式回调，`false`=一次性返回 |
| `max_tokens` | 本次最多生成 token；`<=0` 用 `EngineConfig.max_tokens` |

### `FinishReason`

| 取值 | 含义 |
| --- | --- |
| `STOP` | 自然结束（命中 eos），且本轮无 tool_call |
| `TOOL_CALLS` | 自然结束，且本轮产出过 tool_call |
| `LENGTH` | 触达 `max_tokens` 或 `n_ctx` 上限 |
| `CANCELLED` | 被 `Cancel()` 主动打断 |
| `ERROR` | 底层错误（未初始化、编码失败等） |

---

## 版本路线图

| 版本 | 主题 | 状态 |
| --- | --- | --- |
| v0.1 | 主体链路：加载 → 构图 → 前向 → 贪心解码 | ✅ 已完成 |
| v0.2 | 自研 BPE Tokenizer + 增量 KV cache + 计时/内存统计 | ✅ 已完成 |
| v0.3 | 采样策略：temperature / top-k / top-p / 重复惩罚 / 可复现 seed | ✅ 已完成 |
| v0.4 | 量化权重（Q4_K/Q8_0）、CUDA backend | 规划中 |
| **v1.0** | 通用推理接口：`InferenceEngine` + thinking/tool-call 分离 + 无状态多轮 | ✅ 当前版本 |
