# frostfall.llm — Qwen3-ASR 推理

基于 **ggml + C++17** 的本地推理工程。本文介绍当前 **asr/v0.2** 的功能、环境准备和音频转写验证方法。

## 当前功能与使用边界

asr/v0.2 是 **模型主体闭环验证版**：Python 只负责提取参考 Mel 和导出参考答案，C++ 加载 GGUF 后独立完成音频 Encoder、ASR Prompt 构造、分词、特征融合、Decoder prefill 与增量生成，输出转写文字并与 Python 参考逐段对齐。

```text
原始模型 ──转换脚本──→ GGUF ─────────────────────────────┐
                                                        ↓
WAV ── Python 前处理 ──→ Mel ──C++ 音频 Encoder──→ 音频特征 [A,1024]
                          │                              ↓
                          │        C++ Prompt 构造 + 分词 ─┴→ 融合 Embedding
                          │                                      ↓
                          └── Python 参考实现 ──→ 数值参考 ──→ C++ Decoder ──→ 转写文字
```

已提供：
- Qwen3-ASR-0.6B 完整 GGUF 转换和加载，包含 612 个文本/音频权重张量（复用 v0.1 契约）。
- C++ ggml 音频 Encoder：3 层 Conv2d + GELU、展平、conv_out 投影、局部正弦位置编码、18 层音频 Transformer、输出投影。
- C++ ASR Prompt 构造与分词：自动语言 / 指定 `language Chinese<asr_text>` 两种模板，按音频位置数展开 `<|audio_pad|>`。
- C++ 特征融合：audio_pad 位置的文本 Embedding 被音频特征替换，其余模板标记保持文本向量。
- 六段数值对齐：音频塔 conv_out / 逐层 hidden / 投影 features / Prompt IDs / 融合 Embedding / prefill·teacher forcing·自由生成。

**尚未提供直接接收 PCM/WAV 的 C++ ASR API。**C++ Log-Mel、语言/正文结构化返回（独立 `language` 字段）、流式、取消和跨窗口长音频（超过约 8 秒）属后续版本。当前程序不是麦克风实时识别服务，也不承诺准确率或实时性能。

## 运行一条音频

### 1. 转换模型，只需做一次

```bash
mkdir -p models
python tools/convert_asr_hf_to_gguf.py "$MODEL_DIR" \
    --outfile models/qwen3-asr-0.6b-f16.gguf \
    --mapping-out doc/asr/asr_tensor_mapping.json
```

关键输出为 `mapping OK: 612 tensors (text=311, audio=301)`。生成 `models/qwen3-asr-0.6b-f16.gguf`（C++ 使用的权重、配置及词表）与 `doc/asr/asr_tensor_mapping.json`（张量映射清单）。

已有同一原始模型转换出的 GGUF 可跳过。换音频不需要重新转换模型；GGUF 与参考数据必须来自同一模型版本。

### 2. 导出这条音频的参考 Mel 与参考答案

```bash
python scripts/asr/export_asr_reference.py "$MODEL_DIR" \
    --wav data/test_asr_001.wav \
    --out-dir work/asr_ref_v02 \
    --max-new-tokens 128
```

替换 `--wav` 即可使用自己的录音。建议单声道 16 kHz PCM16 WAV 且**不超过 8 秒**（v0.2 单注意力窗口约束，A ≤ 104；超长会被 C++ 拒绝）；这是工程边界，不是模型能力上限。Python WAV 入口支持 8/16/32-bit 整数 PCM，多声道取平均，非 16 kHz 线性插值重采样；24-bit、浮点 WAV、MP3 请预先转换。

脚本自动生成两套案例目录：

| 目录 | 模式 | 内容 |
| --- | --- | --- |
| `work/asr_ref_v02/auto/` | 自动语言 | 模型自行生成 `language X<asr_text>` 协议头和正文 |
| `work/asr_ref_v02/lang/` | 指定中文 | 输入已含 `language Chinese<asr_text>`，继续生成正文 |

每个案例包含 14 个 `.npy`、`test.wav`、`meta.json`。相比 v0.1 新增的音频中间量是 C++ 音频 Encoder 的对齐答案：

| 文件 | 形状 | 用途 |
| --- | --- | --- |
| `mel.npy` | `[128,T]` | **C++ 音频 Encoder 的实际输入** |
| `audio_conv_out.npy` | `[A,896]` | CNN + 位置编码输出（去 padding 后） |
| `audio_hidden.npy` | `[19,A,896]` | 音频 Transformer 逐层输出 |
| `audio_features.npy` | `[A,1024]` | 音频塔最终投影输出 |
| `ids.npy` / `embd.npy` / `decode_ids.npy` / `decode_logits.npy` | — | Prompt / 融合 / 生成参考（对照 C++ 自己构造的结果） |

默认目录行为：不传 `--wav` 时写入 `work/asr_ref`（内置 5 秒合成信号）；传 `--wav` 且未显式指定 `--out-dir` 时自动切到 `work/asr_ref_wav`，避免覆盖合成基线。建议每次换音频使用新的 `--out-dir`。**v0.1 旧参考目录没有 `audio_*` 中间量，v0.2 工具在读参考数组阶段即抛错终止（提示缺失文件名），需用当前脚本重新导出。**

### 3. 编译 ASR 程序

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFROSTFALL_BUILD_EXAMPLES=ON
cmake --build build --target asr_align_v02 -j4
```

`-j4` 为编译并行度，可按机器内存和 CPU 调整。产物：

```text
build/libfrostfall.so
build/examples/asr/api_test/asr_align_v02
```

### 4. C++ 转写这条音频并检查对齐

```bash
# 自动语言：模型自己输出协议头 + 正文
./build/examples/asr/api_test/asr_align_v02 \
    --gguf models/qwen3-asr-0.6b-f16.gguf \
    --ref-dir work/asr_ref_v02/auto \
    --mode auto \
    --max-gen 160

# 指定中文：输入已含 language Chinese<asr_text>
./build/examples/asr/api_test/asr_align_v02 \
    --gguf models/qwen3-asr-0.6b-f16.gguf \
    --ref-dir work/asr_ref_v02/lang \
    --mode lang \
    --language Chinese \
    --max-gen 160
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--ref-dir` | 指向具体 `auto/` 或 `lang/` 案例，不是父目录 |
| `--mode` | `auto` 或 `lang`，须与参考目录的案例一致 |
| `--language` | lang 模式的目标语言，须与参考导出一致（当前脚本固定 `Chinese`） |
| `--context` | 附加 system 上下文（可选；参考数据默认为空，改了会导致 ids 对不上） |
| `--max-gen` | 自由生成上限，默认 32，也影响 KV 容量预留；应覆盖参考 `n_gen` |

生成上限语义：Python `--max-new-tokens` 是参考生成循环上限（循环结束还会多保存一步，`n_gen` 可能达上限 + 1）；C++ `--max-gen` 是自由生成上限。**参考序列应自然生成到 EOS**——若 Python 达到上限仍未结束，请提高其上限并重新导出，同时提高 C++ 上限；单纯放大 C++ 上限无法补全已截断的参考。

## 如何看结果与排查问题

示意输出（具体数值和文字随音频、模型及环境变化）：

```text
== asr/v0.2 alignment ==
  mode=auto T(mel)=190 A_ref=25 S_ref=40 n_gen_ref=8
-- audio encoder --
  A(cpp)=25 (ref 25)
  [conv_out] n=22400 ... rmse=0.00035
  [audio_layer 0..17] ...（18 层逐层误差）
  [audio_features] ... rmse=4.0e-05
-- prompt / tokenize --
  S(cpp)=40 (ref 40)
  [ids] match
  audio_pad count=25 (A=25)
-- fusion --
  [fused_embd] ... rmse=3.2e-05
-- prefill --
  [prefill_logits] ... top10_hit=10/10
-- teacher forcing (7 steps) --
  [decode_logits] worst_max_abs=... worst_top10_hit=10/10
-- free greedy generation --
  [free_gen] n=8 (ref 8) prefix_match=8/8
  [raw ] language Chinese<asr_text>今天天气怎么样？<|im_end|>
  [text] language Chinese今天天气怎么样？
== result: PASS (failures=0) ==
```

- 六段检查按依赖链排布：**音频塔（conv_out → 逐层 → features）通过了，ids 对比才有意义；ids 通过了，融合对比才有效**。定位问题时先看最先异常的段。
- `[raw]` 保留协议头与 special tokens；`[text]` 为去掉 special 的正文。auto 模式正文以 `language ...<asr_text>` 协议头开头（尚未拆成独立字段，结构化解析属 v0.3）；lang 模式协议头在输入里，正文直接是转写文字。
- `PASS` / 退出码 0：本案例通过工具内置数值检查；不是 CER/WER 准确率评估，也不是完整输入健壮性保证。`FAIL` / 非零退出：按下表排查，不要仅凭文字相似判定通过。

| 现象 | 优先检查 |
| --- | --- |
| Python 提示缺少模块 | 是否 `conda activate cv`；执行前面的 import 检查 |
| 找不到模型文件、架构或张量数不符 | `MODEL_DIR` 是否指向原始完整模型，GGUF 是否由本工程 ASR 转换脚本生成 |
| 找不到 `audio_*` npy（程序终止并提示缺失文件名） | 参考目录是否为旧版 v0.1 导出（无音频中间量），需用当前脚本重新导出 |
| `[ids] MISMATCH` / S 不一致 | `--mode` / `--language` / `--context` 是否与参考案例一致；Prompt 中的 special token 名须与词表一致 |
| `A mismatch` 或音频编码器报超长 | mel 维度、音频时长是否超出单窗口（v0.2 上限约 8 秒 / T ≤ 800） |
| 音频塔误差大（接近或超过阈值） | 分块 padding、位置编码、mel 与 `audio_*` 是否同一轮导出；再按 [代码分析 §4](doc/asr/code_analysis_asr-v0.2.md) 的分段归属定位 |
| 生成文字相同但长度比较失败 | 两端生成上限、参考是否自然到 EOS、是否混用旧参考目录 |
| 数值对齐但文字不符合录音 | 输入音质与内容、模型识别能力；数值对齐与识别质量是两件事 |

没有真实音频时，可以使用内置的确定性 5 秒合成信号（T=500，5 个完整 CNN 块，A=65）检查计算链路：

```bash
python scripts/asr/export_asr_reference.py "$MODEL_DIR" --out-dir work/asr_ref
./build/examples/asr/api_test/asr_align_v02 \
    --gguf models/qwen3-asr-0.6b-f16.gguf --ref-dir work/asr_ref/auto --max-gen 64
./build/examples/asr/api_test/asr_align_v02 \
    --gguf models/qwen3-asr-0.6b-f16.gguf --ref-dir work/asr_ref/lang --max-gen 64
```

合成信号不是人声，输出文字不代表识别率。
