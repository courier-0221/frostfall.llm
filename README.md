# frostfall.llm

> 基于 **ggml v0.15.3** 的极简教学版大模型推理框架，唯一目标模型：**Qwen3-0.6B**。
>
> 设计目标：用尽量少、尽量清晰的 C++ 代码，把"加载模型 → 分词 → 构图 → 前向 → 增量解码"的完整链路跑通，
> 替代 llama.cpp 用于原理学习。详细设计见 [doc/design.md](doc/design.md)。

---

## 当前版本：v0.3

v0.3 在 v0.2（自研分词 + 增量 KV cache + 计时统计）的基础上新增了**采样策略**：

- **采样模块** `src/sampler.{h,cpp}`：支持一条常见的采样链 `重复惩罚 → top-k → temperature+softmax → top-p → 按概率随机采样`，每一步都可单独关闭。
- **可复现随机采样**：可通过 `--seed` 固定随机种子；不指定时运行时随机取一个并打进日志，便于复现。
- **默认即贪心**：不带采样参数时 `--temp 0`，退化为贪心 argmax，输出与 v0.2 逐 token 一致（对拍验证不受影响）。

v0.2 的自研 Tokenizer、增量 KV cache、计时/内存统计、`-i`/`-o`（token id 文件）对拍接口均保留不变。

> v0.2 的功能说明详见下文与 [doc/code_analysis_v0.2.md](doc/code_analysis_v0.2.md)；v0.3 采样实现的逐接口走读见 [doc/code_analysis_v0.3.md](doc/code_analysis_v0.3.md)。

---

## 目录结构

```
frostfall.llm/
├── src/
│   ├── main.cpp          # CLI 入口，加载 -> 分词 -> prefill -> decode 主循环
│   ├── common.{h,cpp}    # 计时器、chat 模板、字节数格式化
│   ├── model.{h,cpp}     # GGUF 加载，qwen3_model 权重结构体
│   ├── graph.{h,cpp}     # qwen3_build_graph()，搭出 Qwen3 前向计算图
│   ├── kv_cache.{h,cpp}  # 增量 KV cache
│   ├── tokenizer.{h,cpp} # 自研 byte-level BPE 分词器（从 GGUF 元数据构建）
│   └── sampler.{h,cpp}   # 采样策略（greedy / temperature / top-k / top-p / 重复惩罚）
├── scripts/
│   ├── encode_prompt.py     # 文本 -> token id 文件（仅供对拍脚本使用，依赖 transformers）
│   ├── decode_tokens.py     # token id 文件 -> 文本（仅供对拍脚本使用，依赖 transformers）
│   └── compare_llamacpp.py  # 与 llama.cpp 对拍验证
├── tools/                # gguf 转换脚本
├── third_party/ggml/     # ggml v0.15.3（内置源码，随仓库跟踪）
├── doc/design.md         # 完整设计文档
├── doc/code_analysis_v0.2.md  # v0.2 逐接口代码走读
├── doc/code_analysis_v0.3.md  # v0.3 采样策略逐接口走读
└── CMakeLists.txt
```

---

## 环境依赖

推理本身（构建 + 运行 `frostfall`）只需要 C++ 工具链，**不再需要 Python**：

| 依赖 | 说明 |
| --- | --- |
| CMake ≥ 3.14 | 构建系统 |
| C++17 编译器 | GCC 9+ 或 Clang 10+ |

以下依赖仅在做 HF → GGUF 转换 或 与 llama.cpp 对拍验证 时才需要：

| 依赖 | 说明 |
| --- | --- |
| Python 3.8+ / `transformers` | 仅用于 `tools/convert_hf_to_gguf.py` 转换模型，以及对拍脚本 `scripts/compare_llamacpp.py` |
| Qwen3-0.6B HF 模型 | 转换为 GGUF 后推理，或对拍时提供"标准答案" tokenizer |

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

推理需要 F16 GGUF 格式，用 llama.cpp 的转换脚本生成（GGUF 中会内置 tokenizer 元数据，v0.2 推理时无需再依赖 HF 模型目录）：

```bash
python tools/convert_hf_to_gguf.py \
    /path/to/Qwen3-0.6B \
    --outfile models/qwen3-0.6b-f16.gguf \
    --outtype f16
```

---

## 使用（一条命令直接推理）

`-p` 直接传文本即可，默认套用 Qwen3 chat 模板；生成结果逐 token 流式打印到 stdout。
默认走**贪心解码**（`--temp 0`），输出确定、可与 llama.cpp 对拍。

```bash
./build/frostfall \
    -m models/qwen3-0.6b-f16.gguf \
    -p "你好，介绍一下你自己" \
    -n 128 \
    -t 4
```

其他常用用法：

```bash
# 不套 chat 模板，直接对纯文本编码（对拍脚本用）
./build/frostfall -m models/qwen3-0.6b-f16.gguf -p "Hello" -n 32 --no-chat-template

# 开启 Qwen3 思考模式（<think> ... </think>）
./build/frostfall -m models/qwen3-0.6b-f16.gguf -p "1+1等于几" -n 128 --think

# 限制 KV cache 上下文长度（默认 4096）
./build/frostfall -m models/qwen3-0.6b-f16.gguf -p "你好" -n 64 -c 2048
```

### 采样（v0.3 新增）

默认 `--temp 0` 为贪心解码。传入采样参数即可开启随机采样，让输出更有多样性：

```bash
# temperature + top-k + top-p 采样（典型 Qwen3 采样配置）
./build/frostfall -m models/qwen3-0.6b-f16.gguf -p "写一首关于秋天的短诗" -n 128 \
    --temp 0.8 --top-k 20 --top-p 0.95

# 固定随机种子 -> 输出可复现（同一 seed 每次结果一致）
./build/frostfall -m models/qwen3-0.6b-f16.gguf -p "讲个笑话" -n 128 \
    --temp 0.8 --top-k 20 --top-p 0.95 --seed 42

# 开启重复惩罚，抑制近期重复的 token
./build/frostfall -m models/qwen3-0.6b-f16.gguf -p "介绍一下你自己" -n 128 \
    --temp 0.8 --repeat-penalty 1.1 --repeat-last-n 64
```

采样参数说明：

| 参数 | 含义 | 默认 | 关闭取值 |
| --- | --- | --- | --- |
| `--temp` | 采样温度，越大越随机；`<=0` 退化为贪心 argmax | `0` | `<=0`（贪心） |
| `--top-k` | 只在 logit 最高的 N 个候选里采样 | `0` | `<=0` |
| `--top-p` | nucleus 采样：保留累计概率达到 P 的最小候选集 | `1.0` | `>=1.0` |
| `--seed` | 随机种子，固定后输出可复现；不指定则每次随机（真实种子会打进日志） | 随机 | — |
| `--repeat-penalty` | 重复惩罚系数，`>1` 抑制近期出现过的 token | `1.0` | `1.0` |
| `--repeat-last-n` | 重复惩罚回看窗口；`<0` 表示整段历史 | `64` | — |

> `--temp 0`（默认）时其余采样参数被忽略，输出与 v0.2 逐 token 一致，因此对拍验证不受影响。

进度、计时统计（加载/prefill/decode 耗时、tokens/s）和内存占用打印到 stderr，生成文本流式打印到 stdout。

### 兼容模式：token id 文件输入输出（供对拍脚本使用）

仍可用 `-i`/`-o` 跳过自带分词器，直接读写 token id 文件（配合 `scripts/encode_prompt.py` / `scripts/decode_tokens.py`）：

```bash
python scripts/encode_prompt.py \
    --model /path/to/Qwen3-0.6B \
    --prompt "你好，介绍一下你自己" \
    --out logs/tokens_in.txt

./build/frostfall \
    -m models/qwen3-0.6b-f16.gguf \
    -i logs/tokens_in.txt \
    -o logs/tokens_out.txt \
    -n 128 -t 4

python scripts/decode_tokens.py \
    --model /path/to/Qwen3-0.6B \
    --ids-file logs/tokens_out.txt
```

---

## CLI 参数说明

```
frostfall v0.3 - 基于 ggml 的 Qwen3-0.6B 教学版推理框架（自研分词 + 增量 KV cache + 采样策略）

usage: frostfall -m <model.gguf> (-p <prompt> | -i <tokens_in.txt>) [options]

  -m, --model        FILE  Qwen3 GGUF 模型路径（必填）
  -p, --prompt       TEXT  用户输入文本（自研分词器编码；默认套 Qwen3 chat 模板）
  -i, --tokens-in    FILE  改为读入空白分隔的 token id（兼容对拍脚本，与 -p 二选一）
  -o, --tokens-out   FILE  把完整 token id 序列（prompt+生成）写入此文件（可选）
  -n, --n-predict    N     最多生成多少个新 token（默认 64）
  -c, --ctx-size     N     KV cache 上下文上限（默认 4096）
  -t, --threads      N     CPU 计算线程数（默认 4）
      --no-chat-template   不套 chat 模板，直接对 -p 的纯文本编码
      --think              启用 Qwen3 思考模式（默认关闭）

 采样选项（默认贪心解码）：
      --temp         F     采样温度，<=0 表示贪心 argmax（默认 0）
      --top-k        N     只在 logit 最高的 N 个候选里采样，<=0 关闭（默认 0）
      --top-p        F     nucleus 采样累计概率阈值，>=1 关闭（默认 1.0）
      --seed         N     随机种子，可复现采样；不指定则每次随机
      --repeat-penalty F   重复惩罚系数，1.0 关闭（默认 1.0）
      --repeat-last-n  N   重复惩罚回看窗口，<0 表示整段历史（默认 64）
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
| v0.1 | 主体链路：加载 → 构图 → 前向 → 贪心解码 + 对拍验证 | ✅ 已完成 |
| v0.2 | 自研 BPE Tokenizer + 增量 KV cache + 计时/内存统计 | ✅ 已完成 |
| **v0.3** | 采样策略：temperature / top-k / top-p / 重复惩罚 / 可复现 seed | ✅ 当前版本 |
| v0.4 | 量化权重（Q4_K/Q8_0）、CUDA backend（选做） | 规划中 |
| v1.0 | 通用api接口 | 规划中 |
