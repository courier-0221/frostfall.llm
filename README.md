# frostfall.llm — Qwen3-ASR 推理

基于 **ggml + C++17** 的本地推理工程。本文介绍当前 **asr/v0.1** 的功能、环境准备和音频转写验证方法。

## 当前功能与使用边界

asr/v0.1 是 **Decoder 数值对齐验证版**：Python 处理音频并导出融合 Embedding，C++ 加载 GGUF 后独立执行 Decoder，生成 token、显示原始转写输出，并与 Python 参考结果比较。

```text
原始模型 ──转换脚本──→ GGUF ──────────────────────┐
                                                ↓
WAV ──Python 前处理 / 音频编码 / 融合──→ Embedding ──C++ Decoder──→ token / 文字
                                     └──Python Decoder──→ 数值参考 ↗ 对比
```

已提供：
- Qwen3-ASR-0.6B 完整 GGUF 转换和加载，包含 612 个文本/音频权重张量。
- 真实 WAV 或内置合成信号的 Python 参考导出，自动语言和指定中文两套案例。
- C++ 外部 Embedding prefill、KV cache 增量 decode、贪心生成和末位 LM Head。
- 逐层 hidden、logits/top-k、参考 token 历史下的 decode 及自由生成一致性检查。

**尚未提供直接接收 PCM/WAV 的 C++ ASR API。**C++ 音频 Encoder、Log-Mel、语言/正文结构化返回、流式、取消和长音频能力属于后续版本。当前程序不是麦克风实时识别服务，也不承诺准确率或实时性能。

## 环境与模型准备

以下命令均在**仓库根目录**执行；所有相对路径按当前工作目录解析。

| 环节 | 需要的环境 |
| --- | --- |
| 编译、运行 C++ | CMake ≥ 3.14、支持 C++17 的编译器；ggml 和 JSON 依赖已放在 `third_party/` |
| 模型转换 | Python、`torch`、`numpy`、`safetensors`；GGUF writer 使用内置 `tools/gguf-py` |
| 音频参考导出 | 同上，加 `transformers==4.57.6`；参考计算使用 CPU fp32，不要求 CUDA |

建议使用独立 Python 环境（Python 3.10+，优先 3.12）。下面是安装方式，不代表任意依赖组合均已完成数值验收；正式复现应固定实际使用的版本。

```bash
python3 -m venv .venv-asr
source .venv-asr/bin/activate
python -m pip install torch numpy safetensors "transformers==4.57.6"
python -c "import torch, transformers, numpy, safetensors; print(torch.__version__, transformers.__version__, numpy.__version__, safetensors.__version__)"
```

已有环境可直接激活并执行最后一条检查，不需要重新创建。仅运行已有 GGUF + 参考数组的 C++ 程序时不需要 Python；换一条音频仍需运行 Python 导出。

从 [模型资源文档](doc/qwen_model/Qwen3-ASR-0.6B_Resource.md) 中的 ModelScope/Hugging Face 入口获取 **原始 Qwen3-ASR-0.6B**，不要混用其他模型或 `-hf` 变体。将 `MODEL_DIR` 改为实际目录：

```bash
MODEL_DIR="/absolute/path/to/Qwen3-ASR-0.6B"
```

目录需要完整的权重、配置和 Tokenizer 文件，包括：

```text
model.safetensors
config.json
preprocessor_config.json
generation_config.json
chat_template.json
tokenizer_config.json
tokenizer.json / vocab.json / merges.txt 等 Tokenizer 资源
```

脚本当前直接读取单文件 `model.safetensors`，不是分片权重入口。F16 GGUF 约 1.75 GiB；Python 导出还需原始权重、fp32 模型和中间数组的内存，实际峰值显著高于 GGUF 文件大小。

## 运行一条音频

### 1. 转换模型，只需做一次

```bash
mkdir -p models
python tools/convert_asr_hf_to_gguf.py "$MODEL_DIR" \
    --outfile models/qwen3-asr-0.6b-f16.gguf \
    --mapping-out doc/asr/asr_v01_tensor_mapping.json
```

关键输出为 `mapping OK: 612 tensors (text=311, audio=301)`。生成：
- `models/qwen3-asr-0.6b-f16.gguf`：C++ 使用的模型权重、配置及词表。
- `doc/asr/asr_v01_tensor_mapping.json`：原始张量到 GGUF 的映射清单。

已有同一原始模型转换出的 GGUF 可跳过。换音频不需要重新转换模型；GGUF 与参考数据必须来自同一模型版本。

### 2. 导出这条音频的输入与参考结果

```bash
python tools/export_asr_reference.py "$MODEL_DIR" \
    --wav data/test_asr_001.wav \
    --out-dir work/asr_ref_wav \
    --max-new-tokens 128
```

替换 `--wav` 即可使用自己的录音。建议先选单声道 16 kHz PCM16 WAV，且不超过 8 秒，以单注意力窗口内的短音频建立对齐基线；这不是脚本已严格实施的时长上限，也不是模型能力上限。

Python WAV 入口支持 8/16/32-bit 整数 PCM，多声道取平均，非 16 kHz 使用线性插值重采样；24-bit、浮点 WAV、MP3 等请预先转换。优先直接提供 16 kHz 波形，避免简单重采样影响质量。

脚本自动生成两套目录：

| 目录 | 模式 | 内容 |
| --- | --- | --- |
| `work/asr_ref_wav/auto/` | 自动语言 | 模型自行生成语言协议和正文 |
| `work/asr_ref_wav/lang/` | 指定中文 | 输入已含 `language Chinese<asr_text>`，继续生成正文 |

每套包含 11 个 `.npy`、`test.wav`、`meta.json`。`meta.json` 保存 Prompt、音频长度和生成长度；`gen_preview` 只展示前几个 token，不是完整转写。当前 CLI 没有任意 `--language` 或 `--context` 参数。

### 3. 编译 ASR 程序

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFROSTFALL_BUILD_EXAMPLES=ON
cmake --build build --target asr_align_v01 -j2
```

`-j2` 为编译并行度，可按机器内存和 CPU 调整。主要产物：

```text
build/libfrostfall.so
build/examples/asr/api_test/asr_align_v01
```

### 4. C++ 生成文字并检查对齐

```bash
./build/examples/asr/api_test/asr_align_v01 \
    --gguf models/qwen3-asr-0.6b-f16.gguf \
    --ref-dir work/asr_ref_wav/auto \
    --max-gen 160

./build/examples/asr/api_test/asr_align_v01 \
    --gguf models/qwen3-asr-0.6b-f16.gguf \
    --ref-dir work/asr_ref_wav/lang \
    --max-gen 160
```

`--ref-dir` 必须指向具体 `auto/` 或 `lang/`，不是它们的父目录。程序先验证 prefill 和 teacher forcing，再重置有效 KV 历史自由生成，最后打印 `[text]` 与 `PASS/FAIL`。

两个生成上限不同：
- Python `--max-new-tokens`：参考生成循环上限，默认 32；当前实现循环结束后还会额外保存一步，`n_gen` 可能达到上限加 1。
- C++ `--max-gen`：自由生成上限，默认 32，也影响 KV 容量预留。应至少覆盖参考 `n_gen`，上述示例预留为 160。

**参考序列应自然生成到 EOS。**若 Python 达到上限仍未结束，请提高其上限并重新导出，同时提高 C++ 上限。单纯放大 C++ 上限无法补全已经截断的参考序列。

## 如何看结果与排查问题

示意输出（具体数值和文字随音频、模型及环境变化）：

```text
-- prefill (embd entry, layer outputs on) --
  ...逐层 hidden、prefill logits 误差与 top10_hit...
-- teacher forcing (...) --
  ...后续 decode 的最差误差...
-- free greedy generation --
  [free_gen] n=... (ref ...) prefix_match=.../...
  [text] language Chinese<asr_text>识别出的文字。<|im_end|>
== result: PASS (failures=0) ==
```

- `[text]`：C++ 自由生成的原始解码文本，保留协议和 special tokens。指定中文案例通常只有正文和结束标记，因为语言前缀已在输入中。
- `prefix_match`：逐 token 比较结果；通过还要求生成总长度相同。
- `PASS` / 退出码 0：当前案例通过工具内置检查；不是 CER/WER 准确率评估，也不是完整输入健壮性保证。
- `FAIL` / 非零退出：数值、序列不一致或运行失败；不要仅凭文字相似判定通过。

| 现象 | 优先检查 |
| --- | --- |
| Python 提示缺少模块 | 是否激活正确环境；执行前面的 import 检查 |
| 找不到模型文件、架构或张量数不符 | `MODEL_DIR` 是否指向原始完整模型，GGUF 是否由本工程 ASR 转换脚本生成 |
| 找不到 npy | 是否先导出，`--ref-dir` 是否指向具体案例，是否在仓库根目录运行 |
| 生成文字相同但长度比较失败 | 两端生成上限、参考是否自然到 EOS、是否混用了旧参考目录 |
| hidden/logits 偏差大或出现 NaN | 模型版本、参考精度、数组是否完整；按代码分析中的数据交接点定位 |
| 输出为空或不合理 | 先听录音、确认格式；数值对齐与识别质量是两件事 |

没有真实音频时，可以使用内置的确定性 5 秒合成信号检查计算链路：

```bash
python tools/export_asr_reference.py "$MODEL_DIR" --out-dir work/asr_ref
./build/examples/asr/api_test/asr_align_v01 \
    --gguf models/qwen3-asr-0.6b-f16.gguf --ref-dir work/asr_ref/auto --max-gen 64
./build/examples/asr/api_test/asr_align_v01 \
    --gguf models/qwen3-asr-0.6b-f16.gguf --ref-dir work/asr_ref/lang --max-gen 64
```

合成信号不是人声，历史基线出现过 `language None<asr_text>` 或 `哦。`，不能用这些文字评价识别率。更换测试音频时建议使用新的 `--out-dir`，避免覆盖已有基线；不传 `--out-dir` 时，真实 WAV 默认写入 `work/asr_ref_wav`，合成信号默认写入 `work/asr_ref`。
