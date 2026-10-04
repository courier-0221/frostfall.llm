# frostfall.llm — Qwen3-ASR 推理

基于 **ggml + C++17** 的本地推理工程。当前版本为 **asr/v0.3**：首个纯 C++ 离线转写版——输入单声道 16 kHz PCM，输出语言与转写文字，全程不依赖 Python 运行时。本文介绍 v0.3 的功能边界、离线转写用法、C++ API 接入方式，以及数值对齐验证方法。

- 版本规划与验收历史：[doc/asr/design_asr.md](doc/asr/design_asr.md)（v0.3 实施记录见 §4.7）
- v0.3 代码分析：[doc/asr/code_analysis_asr-v0.3.md](doc/asr/code_analysis_asr-v0.3.md)
- v0.2 音频塔 / Decoder 计算细节：[doc/asr/code_analysis_asr-v0.2.md](doc/asr/code_analysis_asr-v0.2.md)
- 模型架构：[doc/qwen_model/Qwen3-ASR-0.6B_Model_Architecture_And_Inference.md](doc/qwen_model/Qwen3-ASR-0.6B_Model_Architecture_And_Inference.md)

## 1. 功能与边界

```text
WAV ──/32768──→ PCM ──C++ Log-Mel──→ Mel ──C++ 音频 Encoder──→ 音频特征 [A,1024]
（示例层）                  │                                           ↓
                            │        C++ Prompt 构造 + 分词 ─┴→ 融合 Embedding
                            │                                           ↓
                            └── （仅验证用）golden 参考 ──────→ C++ Decoder ──→ language + text
```

asr/v0.3 在 v0.2（参考 Mel → 音频塔 + Decoder）的基础上补齐了最上游的音频前处理和面向调用方的 API：

- **C++ Log-Mel 音频前处理**（`src/log_mel.*`）：纯 DSP、无权重。Slaney filterbank + 周期 Hann + 居中 STFT（反射填充）+ `log10`/`max-8`/`(x+4)/4` 归一化，逐点对齐官方 `WhisperFeatureExtractor`；参数从 GGUF `asr.mel.*` 契约读取，不硬编码。
- **最小阻塞 ASR API**（`src/asr_engine.*`）：`Init` / `Transcribe`。请求携带 PCM、采样率、context、language、生成上限；响应独立返回 `text`、`language`、`finish_reason`（`eos` / `length` / `context_full`），可附加 mel、T/A/S、生成 token 等调试量。
- **音频 Encoder / Decoder**（复用 v0.2）：3 层 Conv2d + 18 层音频 Transformer + 输出投影；音频特征替换 `audio_pad` 位置的文本 Embedding，贪心解码到 EOS 集合。
- **每请求独立状态**：入口重置 KV，重复调用不串状态；音频塔每请求只跑一次，生成阶段增量解码。

**输入边界（工程边界，非模型能力上限）**：单请求、单音频、**不超过 8 秒**（单注意力窗口约束，A ≤ 104）。核心接口只接收单声道 16 kHz float32 PCM；示例层只支持单声道 16 kHz 16-bit 非压缩 PCM WAV。空输入、非 16 kHz、含 NaN/Inf、超长一律显式报错，**不改采样率标签、不静默截断**。

文本流式、取消、长音频（跨窗口）、麦克风持续输入、GPU、量化属后续版本；不承诺识别准确率（CER/WER 回归属 v1.0）或实时性能。本工程同时保留 LLM 聊天能力（`Frostfall::InferenceEngine`，示例在 `examples/llm/api_test/`），ASR 与 LLM 共享 Decoder 计算但接口相互独立。

## 2. 环境准备与构建

```bash
# 编译（cmake >= 3.14，C++17；仅推理不需要 Python）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFROSTFALL_BUILD_EXAMPLES=ON
cmake --build build -j4
```

构建产物：

```text
build/libfrostfall.so                             核心库（model/graph/log_mel/asr_engine/...）
build/examples/asr/api_test/asr_transcribe_v03    v0.3 离线转写示例（本文主流程）
build/examples/asr/api_test/asr_align_v02         v0.2 数值对齐工具（回归用）
```

模型转换只需一次（需要 Python，`conda activate cv`）：

```bash
mkdir -p models
python tools/convert_asr_hf_to_gguf.py "$MODEL_DIR" \
    --outfile models/qwen3-asr-0.6b-f16.gguf \
    --mapping-out doc/asr/asr_tensor_mapping.json
```

关键输出 `mapping OK: 612 tensors (text=311, audio=301)`，生成 `models/qwen3-asr-0.6b-f16.gguf`（权重、超参、词表、`asr.mel.*` 前处理契约单文件）。已有同一原始模型转换的 GGUF 可跳过；换音频不需要重新转换模型。

## 3. 快速开始：离线转写一条音频

```bash
./build/examples/asr/api_test/asr_transcribe_v03 \
    --config build/examples/asr/api_test/asr_conf_v03.json \
    --wav data/test_asr_001.wav
```

输出（示意；数值随音频与机器变化）：

```text
[transcribe#1] finish=eos language=Chinese text=今天天气怎么样？
  [debug] T=190 A=25 S=40 n_gen=8 log_mel_max=1.34983
  [raw ] language Chinese<asr_text>今天天气怎么样？<|im_end|>
[transcribe#2] finish=eos language=Chinese text=今天天气怎么样？
  [repeat] identical to first call  OK
[with-context] ok=1 finish=eos text=今天天气怎么样？
```

字段解读：

| 字段 | 含义 |
| --- | --- |
| `finish=eos` | 正常结束：生成到 EOS 集合（`finish=length` 达到生成上限；`finish=context_full` KV 容量耗尽） |
| `language=Chinese` | 指定模式返回请求值；自动模式返回模型协议头声明的语言（无声内容时可能为 `None`） |
| `text=...` | 转写正文：已剥离 `language X` 协议头、`<asr_text>` 标记与所有 special token |
| `T/A/S/n_gen` | 有效 Mel 帧数 / 音频位置数 / prefill 长度 / 生成 token 数（长度链由 T 逐级推导） |
| `log_mel_max` | 归一化后 Log-Mel 矩阵最大值（`= (log10_max+4)/4`，C++ 与 golden 一致为好） |
| `[repeat] ... OK` | 同一输入第二次调用的验证：逐 token 与第一次一致（每请求 KV 重置） |

命令行参数：

| 参数 | 说明 |
| --- | --- |
| `--config` | 引擎 JSON 配置（见 `examples/asr/api_test/asr_conf_v03.json`）；也可用 `--gguf <path>` 直接指定模型 |
| `--wav` | 输入 WAV：只接受单声道 16 kHz 16-bit 非压缩 PCM（立体声 / 其他采样率 / 其他位深显式报错） |
| `--pcm` | 直接输入 float32 PCM 的 `.npy`（16 kHz） |
| `--ref-dir` | 对齐模式：输入参考目录的 `pcm.npy`，对比引擎内部 Log-Mel 与 golden `mel.npy`、生成 token 与 golden `decode_ids.npy`（见 §5） |
| `--language` | 指定语言（如 `Chinese`）；缺省为自动语言 |
| `--context` | 附加 system 上下文（转写风格 / 领域提示） |
| `--max-gen` | 单次转写的生成 token 上限，达到则 `finish=length` |
| `--check-errors` | 演示四类拒绝路径：空输入 / 非 16 kHz / 超过 8 秒 / NaN |

错误示例（`--check-errors`，全部被显式拒绝）：

```text
[empty-input] rejected: empty input: n_samples=0
[wrong-rate ] rejected: unsupported sample_rate=44100 (this engine only accepts 16000 Hz; resample before calling)
[too-long   ] rejected: audio too long: n_samples=144000 (9s) > limit 8s
[nan-input  ] rejected: pcm contains NaN/Inf at sample 15200
```

## 4. 在 C++ 中接入（最小阻塞 API）

```cpp
#include "asr_engine.h"

qwen3_asr_engine engine;
qwen3_asr_engine_config cfg;
cfg.model_path = "models/qwen3-asr-0.6b-f16.gguf";  // 其余字段用默认值；也可 InitFromJsonFile 读 JSON
if (!engine.Init(cfg)) return 1;

std::vector<float> pcm;  // 单声道 16 kHz float32（通常已在 [-1,1]）
qwen3_asr_request req;
req.pcm = pcm.data();
req.n_samples = (int32_t) pcm.size();
req.sample_rate = 16000;
req.want_debug = false;         // true 时响应附 mel / T/A/S / gen_ids / raw

const qwen3_asr_response r = engine.Transcribe(req);
if (!r.ok) {
    fprintf(stderr, "error: %s\n", r.error.c_str());
} else {
    printf("[%s] %s\n", r.language.c_str(), r.text.c_str());
}
```

链接 `frostfall` 共享库即可（头文件都在 `src/`）。接口约定：

- **线程模型**：`Init` / `Transcribe` 互斥串行（与 LLM `Frostfall::InferenceEngine` 对齐）；v0.3 无回调、无取消入口。
- **请求字段**：`pcm` / `n_samples` / `sample_rate` / `context` / `language` / `max_gen_tokens`（≤0 用引擎默认）。
- **响应字段**：`ok` / `error` / `text` / `language` / `finish_reason`，`want_debug=true` 时另有 `T` / `A` / `S` / `n_gen` / `gen_ids` / `raw` / `mel` / `log_mel_max`。
- **校验先于计算**：所有输入校验（空 / 采样率 / 超长 / NaN / 容量）在任何计算前完成，失败只返回 `error`，无副作用。
- **每请求独立**：入口重置 KV；同一输入重复调用结果逐 token 一致。

引擎配置字段（`asr_conf_v03.json`）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `model_path` | 必填 | `qwen3-asr` 架构 GGUF 路径 |
| `n_ctx` | `0` | KV 容量；0 = `min(4096, n_ctx_train)` |
| `max_audio_seconds` | `8` | 单请求音频时长上限（秒）；8 秒 ⇔ A ≤ 104 单窗口上限 |
| `max_gen_tokens` | `64` | 请求未指定生成上限时的默认值 |

## 5. 验证：Log-Mel 与全链路 golden 对齐

验证需要 Python（`conda activate cv`）；产品推理链路本身不需要。参考数据由 `scripts/asr/export_asr_reference.py` 一次导出，C++ 之后离线复用。

### 5.1 导出 golden 参考

```bash
python scripts/asr/export_asr_reference.py "$MODEL_DIR" \
    --wav data/test_asr_001.wav \
    --out-dir work/asr_ref_v02 \
    --max-new-tokens 128
```

自动生成两套案例：`auto/`（模型自行输出 `language X<asr_text>` 协议头）与 `lang/`（输入已预置 `language Chinese<asr_text>`）。每个案例含 `pcm.npy`、`mel.npy`（官方 WhisperFeatureExtractor 的 golden Log-Mel，`[128,T]`）、`decode_ids.npy`（参考生成 token）以及音频塔逐段中间量（供 asr_align_v02 回归）。规范输入为单声道 16 kHz PCM16 WAV 且不超过 8 秒。

### 5.2 运行 v0.3 对齐

```bash
# 自动语言
./build/examples/asr/api_test/asr_transcribe_v03 \
    --config build/examples/asr/api_test/asr_conf_v03.json \
    --ref-dir work/asr_ref_v02/auto

# 指定中文（--language 须与参考导出一致）
./build/examples/asr/api_test/asr_transcribe_v03 \
    --config build/examples/asr/api_test/asr_conf_v03.json \
    --ref-dir work/asr_ref_v02/lang --language Chinese
```

示意输出：

```text
-- golden alignment (work/asr_ref_v02/auto) --
  [log_mel] n=24320 max_abs=2.13385e-05 rmse=6.17421e-07  OK
  [gen_ids] n_cpp=8 n_ref=8 prefix_match=8/8  OK
== result: PASS (failures=0) ==
```

判定口径（工具内置阈值，基线远小于阈值）：

| 检查 | 阈值 | 实测基线（data/test_asr_001.wav，T=190） |
| --- | --- | --- |
| Log-Mel vs golden `mel.npy` | `max_abs < 1e-4` 且 `rmse < 1e-5` | max_abs=2.13e-5，RMSE=6.17e-7（差异源：官方 pocketfft-f32 vs 本实现 double FFT 的量化点） |
| 生成 token vs golden `decode_ids.npy` | 数量与逐 token 完全一致 | auto 8/8，lang 5/5 |
| 重复调用一致性 | 二次调用逐 token 一致 | 一致 |

辅助检查：

- 静音输入：输出 `finish=eos language=None text=<空>`，正常。
- 合成英文语音（4.5s）：返回 `language=English` 与英文标点正文；合成音质下**识别内容不作为验收项**，仅验证链路。
- v0.2 六段对齐工具（音频塔 conv_out / 逐层 / 投影 / ids / 融合 / logits / teacher forcing）保留用于更细的数值定位：

```bash
./build/examples/asr/api_test/asr_align_v02 \
    --gguf models/qwen3-asr-0.6b-f16.gguf \
    --ref-dir work/asr_ref_v02/auto --mode auto --max-gen 160
```

用法细节（`--mode` 须与案例一致、`--max-gen` 应覆盖参考 `n_gen`、参考须自然生成到 EOS 等）见 [v0.2 代码分析](doc/asr/code_analysis_asr-v0.2.md)与工具内 `--help`。

LLM 回归（共享 Decoder 计算未被 ASR 改动破坏）：LLM 四类调用示例默认不构建，需先在根 [CMakeLists.txt](CMakeLists.txt) 打开 `add_subdirectory(examples/llm/api_test)` 后重新构建：

```bash
./build/examples/llm/api_test/infer_nothink_blocking --config build/examples/llm/api_test/llm_infer_conf.json
# 另三类：infer_nothink_stream / infer_yesthink_blocking / infer_yesthink_stream
```

## 6. 排查问题

| 现象 | 优先检查 |
| --- | --- |
| Python 提示缺少模块（转换 / 导出参考时） | 是否 `conda activate cv`；LLM 回归示例与 ASR 无 Python 依赖 |
| 找不到模型文件、架构或张量数不符 | `MODEL_DIR` 是否指向原始完整模型；GGUF 是否由本工程 ASR 转换脚本生成（`general.architecture = qwen3-asr`） |
| WAV 被拒绝（channels / 位深 / 采样率） | 输入需转成单声道 16 kHz 16-bit PCM WAV 后再试；本版不自动重采样或混音 |
| 引擎报 `audio too long ... > limit 8s` | 输入超 8 秒（单窗口约束 A ≤ 104）；截短音频或等待 v0.4 跨窗口 |
| 引擎报 `context insufficient: prefill ... > n_ctx` | `n_ctx` 配置太小；提高 `n_ctx`（受 `n_ctx_train` 截断） |
| `[log_mel] FAIL` / mel 形状不一致 | `--ref-dir` 案例与 `--language` 是否匹配；golden 目录是否用当前脚本对同一音频导出；音频是否与 `pcm.npy` 同源 |
| `[gen_ids] FAIL` 但 `[log_mel]` OK | 生成上限是否覆盖参考 `n_gen`；参考是否自然到 EOS；再跑 asr_align_v02 定位音频塔 / 融合段 |
| Log-Mel 对齐但文字与录音不符 | 数值对齐与识别质量是两件事；检查音频内容与音质，准确率回归属 v1.0 |
| `finish=length` / `context_full` | 不是错误：生成到上限 / KV 耗尽的结构化截断语义，需要的话提高 `--max-gen` 或 `n_ctx` |

## 7. 后续版本

完整离线 API（结构体/JSON 初始化、取消、文本流式、分批 prefill、跨窗口长音频）属 `asr/v0.4`；持续音频输入与可修订的流式转写属 `asr/v0.5`；质量与资源回归、发布契约属 `asr/v1.0`。明细见 [版本规划](doc/asr/design_asr.md)。
