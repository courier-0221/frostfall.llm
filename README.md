# frostfall.llm

> 基于 **ggml v0.15.3** 的极简教学版大模型推理框架，唯一目标模型：**Qwen3-0.6B**。
>
> 设计目标：用尽量少、尽量清晰的 C++ 代码，把"加载模型 → 构图 → 前向 → 贪心解码"的完整链路跑通，
> 替代 llama.cpp 用于原理学习。详细设计见 [doc/design.md](doc/design.md)。

---

## 当前版本：v0.1

v0.1 的两个简化（后续版本会逐步去掉）：

1. **Tokenizer 由 Python 预处理**：用 `scripts/encode_prompt.py` 把文本编成 token id 文件，C++ 端直接读；生成结果同样用 Python 反解码。
2. **无增量 KV cache**：每一步都把当前完整 token 序列重新前向一遍（O(n²)，但最简单、最不易出 bug），增量 KV cache 推迟到 v0.2。

---

## 目录结构

```
frostfall.llm/
├── src/
│   ├── main.cpp          # CLI 入口，推理主循环
│   ├── model.{h,cpp}     # GGUF 加载，qwen3_model 权重结构体
│   └── graph.{h,cpp}     # qwen3_build_graph()，搭出 Qwen3 前向计算图
├── scripts/
│   ├── encode_prompt.py  # 文本 -> token id 文件（依赖 transformers）
│   ├── decode_tokens.py  # token id 文件 -> 文本（依赖 transformers）
│   └── compare_llamacpp.py  # 与 llama.cpp 对拍验证
├── third_party/ggml/     # ggml v0.15.3（git submodule）
├── doc/design.md         # 完整设计文档
└── CMakeLists.txt
```

---

## 环境依赖

| 依赖 | 说明 |
| --- | --- |
| CMake ≥ 3.14 | 构建系统 |
| C++17 编译器 | GCC 9+ 或 Clang 10+ |
| Python 3.8+ | 仅用于 tokenizer 预处理脚本 |
| `transformers` | `pip install transformers` |
| Qwen3-0.6B HF 模型 | 用于 tokenizer；转换为 GGUF 后推理 |

---

## 构建

```bash
git clone --recurse-submodules <repo>
cd frostfall.llm

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 产物
./build/frostfall --help
```

> 如果已克隆但忘了拉 submodule：`git submodule update --init`

---

## 准备模型（HF → GGUF）

推理需要 F16 GGUF 格式，用 llama.cpp 的转换脚本生成：

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py \
    /path/to/Qwen3-0___6B \
    --outfile models/qwen3-0.6b-f16.gguf \
    --outtype f16
```

转换结果放到 `models/`（已加入 `.gitignore`，不会被提交）。

---

## 使用（v0.1 三步流程）

### 第一步：编码 prompt → token id 文件

```bash
# 套用 Qwen3 chat 模板（推荐）
python scripts/encode_prompt.py \
    --model /path/to/Qwen3-0___6B \
    --prompt "你好，介绍一下你自己" \
    --out tokens_in.txt

# 不套 chat 模板，直接编码纯文本（对拍脚本用）
python scripts/encode_prompt.py \
    --model /path/to/Qwen3-0___6B \
    --prompt "Hello" \
    --out tokens_in.txt \
    --no-chat-template

# 开启 Qwen3 思考模式（<think> ... </think>）
python scripts/encode_prompt.py \
    --model /path/to/Qwen3-0___6B \
    --prompt "1+1等于几" \
    --out tokens_in.txt \
    --enable-thinking
```

### 第二步：推理

```bash
./build/frostfall \
    -m models/qwen3-0.6b-f16.gguf \
    -i tokens_in.txt \
    -o tokens_out.txt \
    -n 128 \
    -t 8
```

进度和计时打印到 stderr，完整 token id 序列（prompt + 生成）打印到 stdout 并写入 `-o` 文件。

### 第三步：解码 token id → 文本

```bash
# 查看完整输出（含 prompt）
python scripts/decode_tokens.py \
    --model /path/to/Qwen3-0___6B \
    --ids-file tokens_out.txt

# 只看新生成的部分（需告知 prompt 长度）
python scripts/decode_tokens.py \
    --model /path/to/Qwen3-0___6B \
    --ids-file tokens_out.txt \
    --prompt-len $(wc -w < tokens_in.txt)
```

---

## CLI 参数说明

```
frostfall v0.1 - 基于 ggml 的 Qwen3-0.6B 教学版推理框架

usage: frostfall -m <model.gguf> -i <tokens_in.txt> [-o <tokens_out.txt>] [-n N] [-t N]

  -m, --model       FILE   Qwen3 GGUF 模型路径（必填）
  -i, --tokens-in   FILE   prompt 的 token id 文件，空白分隔的整数（必填）
  -o, --tokens-out  FILE   生成结果（完整 token id 序列）写入此文件（可选）
  -n, --n-predict   N      最多生成多少个新 token（默认 64）
  -t, --threads     N      CPU 计算线程数（默认 4）
```

---

## 正确性验证（对拍）

v0.1 自带与 llama.cpp 的对拍脚本。同一 prompt、`--temp 0`，逐 token 比对：

```bash
python scripts/compare_llamacpp.py \
    --frostfall ./build/frostfall \
    --llama-cli /path/to/llama.cpp/llama-cli \
    --model models/qwen3-0.6b-f16.gguf \
    --hf-model /path/to/Qwen3-0___6B \
    --prompt "你好"
```

---

## 版本路线图

| 版本 | 主题 | 状态 |
| --- | --- | --- |
| **v0.1** | 主体链路：加载 → 构图 → 前向 → 贪心解码 + 对拍验证 | ✅ 当前版本 |
| v0.2 | 自研 BPE Tokenizer + 增量 KV cache + 模块化 | 规划中 |
| v0.3 | 采样策略：temperature / top-k / top-p | 规划中 |
| v0.4 | 量化权重（Q4_K/Q8_0）、CUDA backend（选做） | 规划中 |
