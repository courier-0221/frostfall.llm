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
├── tools/                # gguf 转换脚本
├── third_party/ggml/     # ggml v0.15.3（内置源码，随仓库跟踪）
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
git clone <repo>
cd frostfall.llm

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 产物
./build/frostfall --help
```

---

## 准备模型（HF → GGUF）

推理需要 F16 GGUF 格式，用 llama.cpp 的转换脚本生成：

```bash
python /path/to/convert_hf_to_gguf.py \
    /path/to/Qwen3-0.6B \
    --outfile models/qwen3-0.6b-f16.gguf \
    --outtype f16
```

---

## 使用（v0.1 三步流程）

### 第一步：编码 prompt → token id 文件

```bash
# 套用 Qwen3 chat 模板（推荐）
python scripts/encode_prompt.py \
    --model /path/to/Qwen3-0.6B \
    --prompt "你好，介绍一下你自己" \
    --out logs/tokens_in.txt

# 不套 chat 模板，直接编码纯文本（对拍脚本用）
python scripts/encode_prompt.py \
    --model /path/to/Qwen3-0.6B \
    --prompt "Hello" \
    --out logs/tokens_in.txt \
    --no-chat-template

# 开启 Qwen3 思考模式（<think> ... </think>）
python scripts/encode_prompt.py \
    --model /path/to/Qwen3-0.6B \
    --prompt "1+1等于几" \
    --out logs/tokens_in.txt \
    --enable-thinking
```

### 第二步：推理

```bash
./build/frostfall \
    -m models/qwen3-0.6b-f16.gguf \
    -i logs/tokens_in.txt \
    -o logs/tokens_out.txt \
    -n 128 \
    -t 4
```

进度和计时打印到 stderr，完整 token id 序列（prompt + 生成）打印到 stdout 并写入 `-o` 文件。

### 第三步：解码 token id → 文本

```bash
# 查看完整输出（含 prompt）
python scripts/decode_tokens.py \
    --model /path/to/Qwen3-0.6B \
    --ids-file logs/tokens_out.txt

# 只看新生成的部分（需告知 prompt 长度）
python scripts/decode_tokens.py \
    --model /path/to/Qwen3-0.6B \
    --ids-file logs/tokens_out.txt \
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

### 思路

用成熟的 **llama.cpp 作为「标准答案」**，验证自研的 frostfall 输出是否一致。
在 `--temp 0`（贪心解码、无随机性）前提下，同一模型 + 同一 prompt，每一步都应选出相同 token；
只要两边最终生成的文本一致，就说明 frostfall 的「加载 → 构图 → 前向 → 解码」整条链路是正确的。

### 实现（分 4 步独立执行）

对拍脚本 `scripts/compare_llamacpp.py` 拆成 4 个子命令，各自独立进程运行、结果落盘到 `--workdir`。
这样做的目的是让吃内存的 `transformers` 只在 **encode / compare** 两步加载，跑完即退出释放内存；
`run-ff` 和 `run-llama` 不依赖任何大 Python 包。

| 步骤 | 子命令 | 依赖 transformers | 做的事 |
| --- | --- | --- | --- |
| 1 | `encode` | 是 | 用 HF tokenizer 把 prompt 编成 token id，落盘 `tokens_in.txt` / `prompt_n.txt` / `prompt.txt` |
| 2 | `run-ff` | 否 | 跑 frostfall，输入 token id，输出完整 id 序列到 `tokens_ff.txt` |
| 3 | `run-llama` | 否 | 跑 llama.cpp 补全程序（`--temp 0`；新版用 `llama-completion`，旧版 `llama-cli`），stdout 存到 `llama_out.txt` |
| 4 | `compare` | 是 | 解码 frostfall 新生成的 id 为文本，与 llama.cpp 续写文本比对 |

对比时需对齐两边口径：frostfall 输出是「prompt + 新生成」的完整 token id，脚本切掉前 `prompt_n` 个 id
再解码；llama.cpp 输出的文本含原 prompt 前缀，脚本去掉前缀取续写部分。两段文本 `strip()` 后相等即
「完全一致 ✓」，否则提示按 `doc/design.md §8.5` 用逐层 dump 定位差异。

`workdir` 中的中间文件：

```
tokens_in.txt   prompt token ids（空白分隔整数）
prompt_n.txt    prompt token 数（一个整数）
prompt.txt      prompt 原文
tokens_ff.txt   frostfall 完整输出 token ids（prompt + 新生成）
llama_out.txt   llama-cli stdout 原文
```

### 运行命令

```bash
WORKDIR=work/cmp
HF=/path/to/Qwen3-0.6B
MODEL=models/qwen3-0.6b-f16.gguf

# 步骤 1：编码 prompt（加载 transformers，完成后进程退出，内存释放）
conda run -n llm python scripts/compare_llamacpp.py encode \
    --hf-model $HF --prompt "The capital of France is" --workdir $WORKDIR

# 步骤 2：运行 frostfall（无需 transformers）
python scripts/compare_llamacpp.py run-ff \
    --frostfall-bin build/frostfall --model $MODEL --workdir $WORKDIR -n 16

# 步骤 3：运行 llama.cpp（无需 transformers）
# 注意：新版 llama.cpp（约 b8300+）的 llama-cli 只做交互式聊天、不再支持 -no-cnv，
#       非交互一次性补全请改用 llama-completion；旧版仍可传 llama-cli。
python scripts/compare_llamacpp.py run-llama \
    --llama-cli /path/to/llama.cpp/build/bin/llama-completion \
    --model $MODEL --workdir $WORKDIR -n 16

# 步骤 4：解码并对比（加载 transformers，完成后进程退出）
conda run -n llm python scripts/compare_llamacpp.py compare \
    --hf-model $HF --workdir $WORKDIR
```

---

## 版本路线图

| 版本 | 主题 | 状态 |
| --- | --- | --- |
| **v0.1** | 主体链路：加载 → 构图 → 前向 → 贪心解码 + 对拍验证 | ✅ 当前版本 |
| v0.2 | 自研 BPE Tokenizer + 增量 KV cache + 模块化 | 规划中 |
| v0.3 | 采样策略：temperature / top-k / top-p | 规划中 |
| v0.4 | 量化权重（Q4_K/Q8_0）、CUDA backend（选做） | 规划中 |
| v1.0 | 通用api接口 | 规划中 |
