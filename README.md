# frostfall.llm — Qwen3-ASR 推理

基于 **ggml + C++17** 的本地推理工程。当前版本为 **asr/v0.4**：完整离线 API——输入单声道 16 kHz PCM（默认上限 30 秒，跨注意力窗口），可选择一次取得转写结果，或在完整音频处理后**逐 token 流式回调正文**；支持**取消**、错误分类、分批 prefill 与全链路性能统计，全程不依赖 Python 运行时。

- 版本规划与验收历史：[doc/asr/design_asr.md](doc/asr/design_asr.md)（v0.4 实施记录见 §5.7、跨窗口参考语义见 §5.6）
- v0.4 代码分析：[doc/asr/code_analysis_asr-v0.4.md](doc/asr/code_analysis_asr-v0.4.md)
- v0.3 Log-Mel / 首版 API：[doc/asr/code_analysis_asr-v0.3.md](doc/asr/code_analysis_asr-v0.3.md)
- v0.2 音频塔 / Decoder 计算细节：[doc/asr/code_analysis_asr-v0.2.md](doc/asr/code_analysis_asr-v0.2.md)
- 模型架构：[doc/qwen_model/Qwen3-ASR-0.6B_Model_Architecture_And_Inference.md](doc/qwen_model/Qwen3-ASR-0.6B_Model_Architecture_And_Inference.md)

## 1. 功能与边界

```text
WAV ──/32768──→ PCM ──C++ Log-Mel──→ Mel ──C++ 音频 Encoder──→ 音频特征 [A,1024]
（示例层）                      │        （跨窗口块对角双向注意力）    ↓
                                │        C++ Prompt 构造 + 分词 ─┴→ 融合 Embedding
                                │        （分批 prefill，批边界可取消）    ↓
                                └── （仅验证用）golden 参考 ──→ C++ Decoder ──→ language + text
                                                            （逐 token 贪心，断点可取消 / 可流式回调）
```

asr/v0.4 在 v0.3（PCM → 全 C++ 链路 → 阻塞结果）的基础上补齐应用接入能力：

- **文本流式**（`src/asr_engine.*`）：完整音频处理后逐 token 回调转写正文；`STREAM_LANG` 一次性回调自动语言模式下的语言协议头，`STREAM_TEXT` 按完整 UTF-8 序列边界增量回调正文；语言协议 token 不外泄、不计入转写文字。
- **取消**：`request.cancel`（`std::atomic<bool>*`）由其他线程置位，或流式回调返回 `false`；检查点在音频 CNN 分块批、prefill 批、decode step 边界；取消返回 `finish_reason=cancelled` + 取消时处理阶段 `stage` + 已回调的部分正文。不承诺打断单个 ggml 算子。
- **错误协议**：`resp.err` 程序化分类（`bad_input` / `audio_too_long` / `context_insufficient` / `internal` / `not_init`），`resp.error` 保留人读文本；取消不是错误。
- **分批 prefill**：混合 Embedding 按 `prefill_batch_size` 分批前向（批内位置 / 因果 mask / n_past 同步顺延），末位 logits 与单批逐位一致，峰值计算缓冲下降（S=405/bs=128 实测 6.94 MB）。
- **跨窗口较长音频**：音频 Transformer 按官方 `get_audio_cu_seqlens` 语义做块对角双向注意力（完整块 window = 13×8 = 104 个位置，窗口内双向、窗口间隔离）；单窗口输入（A ≤ 104）与 v0.3 结果逐位一致；音频默认上限放宽至 **30 秒**（A=390，更长输入受 KV 容量约束显式报错）。
- **性能统计**：前处理 / Encoder / prefill / decode 分段耗时、首生成 token、**首可见文字**（不含语言协议 token）、RTF、峰值计算缓冲。
- **Log-Mel / Encoder / Decoder 计算**复用 v0.2~v0.3。

**输入边界（工程边界，非模型能力上限）**：单请求、单音频、不超过 `max_audio_seconds`（默认 30 秒）且 prefill+生成放得进 KV（`n_ctx`）。核心接口只接收单声道 16 kHz float32 PCM；示例层只支持单声道 16 kHz 16-bit 非压缩 PCM WAV。空输入、非 16 kHz、含 NaN/Inf、超长、上下文不足一律显式分类报错，**不改采样率标签、不静默截断**。

本版的"文本流式"是**完整录音处理完成后逐 token 返回文字**，不是边录边识别：持续音频输入与可修订的暂定转写属 `asr/v0.5`；不承诺 20 分钟输入、无界长音频、实时性能或麦克风流式。识别准确率（CER/WER）回归属 `asr/v1.0`。本工程同时保留 LLM 聊天能力（`Frostfall::InferenceEngine`，示例在 `examples/llm/api_test/`），ASR 与 LLM 共享 Decoder 计算但接口相互独立。

## 2. 环境准备与构建

```bash
# 编译（cmake >= 3.14，C++17；仅推理不需要 Python）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFROSTFALL_BUILD_EXAMPLES=ON
cmake --build build -j4
```

构建产物：

```text
build/libfrostfall.so                             核心库（model/graph/log_mel/asr_engine/...）
build/examples/asr/api_test/asr_transcribe_v04    v0.4 完整离线/流式示例（本文主流程）
build/examples/asr/api_test/asr_transcribe_v03    v0.3 最小阻塞示例（回归保留）
```

`asr_align_v01/v02` 数值对齐工具与 LLM 四类回归示例默认不构建：需要时在 [examples/asr/api_test/CMakeLists.txt](examples/asr/api_test/CMakeLists.txt) 与根 [CMakeLists.txt](CMakeLists.txt) 中打开对应的 `add_executable` / `add_subdirectory`（见 §5.4、§5.5）。

模型转换只需一次（需要 Python，`conda activate cv`）：

```bash
mkdir -p models
python tools/convert_asr_hf_to_gguf.py "$MODEL_DIR" \
    --outfile models/qwen3-asr-0.6b-f16.gguf \
    --mapping-out doc/asr/asr_v01_tensor_mapping.json
```

关键输出 `mapping OK: 612 tensors (text=311, audio=301)`，生成 `models/qwen3-asr-0.6b-f16.gguf`（权重、超参、词表、`asr.mel.*` 前处理契约单文件）。已有同一原始模型转换的 GGUF 可跳过；换音频不需要重新转换模型。

## 3. 快速开始：转写、流式与取消

```bash
./build/examples/asr/api_test/asr_transcribe_v04 \
    --config build/examples/asr/api_test/asr_conf_v04.json \
    --wav data/test_asr_001.wav
```

示例对同一输入先跑阻塞模式，再以回调跑文本流式模式，两者最终结果必须一致（输出示意；数值随音频与机器变化）：

```text
[blocking#1] finish=eos stage=done language=Chinese text=今天天气怎么样？
  [stats] total=2.816s rtf=1.482 preproc=0.0129s encoder=0.623s prefill=1.155s (batches=1, bs=40) decode=1.020s (steps=7)
  [stats] first_model_token=1.792s first_visible_text=n/a peak_galloc=1.256MB
[streaming] pieces: <lang:language Chinese> [今天] [天气] [怎么样] [？]
[streaming#1] finish=eos stage=done language=Chinese text=今天天气怎么样？
  [stats] first_model_token=1.746s first_visible_text=2.159s peak_galloc=1.256MB
  [consistency] streamed final == blocking (text bytes 24)  OK
```

字段解读：

| 字段 | 含义 |
| --- | --- |
| `finish=eos` | 正常结束；`length` 达到生成上限；`context_full` KV 容量耗尽；v0.4 新增 `cancelled`（调用方取消，取消不是错误） |
| `stage=done` | 请求结束时的阶段；取消 / 失败时为取消或失败所处阶段（`input/preproc/encoder/prefill/decode`） |
| `language=Chinese` | 指定模式返回请求值；自动模式返回模型协议头声明（无声内容时可能为 `None`） |
| `text=...` | 转写正文：已剥离语言协议头、`<asr_text>` 标记与所有 special token；取消时为已回调的部分正文 |
| `pieces` | 流式增量序列：`<lang:..>` 为一次性语言协议回调，其后的 `[..]` 为按完整 UTF-8 边界回调的正文增量 |
| `rtf` | 端到端处理耗时 / 音频时长（含音频塔与 prefill 的一次性开销，不含录音等待） |
| `first_model_token` ≠ `first_visible_text` | 前者=首生成 token，后者=首个**转写正文**增量；语言协议 token 不计入后者。阻塞模式无回调，`first_visible_text=n/a` |
| `peak_galloc` | Decoder 计算（不含 KV / 权重）峰值分配；`bs` 越小越低 |

命令行参数（v0.4 示例在 v0.3 参数之外新增四个）：

| 参数 | 说明 |
| --- | --- |
| `--config` / `--gguf` | 引擎 JSON 配置（见 `examples/asr/api_test/asr_conf_v04.json`）或直接指定模型 |
| `--wav` / `--pcm` / `--ref-dir` | 输入：WAV（单声道 16 kHz 16-bit PCM）/ float32 `.npy` / 参考目录 `pcm.npy` |
| `--language` / `--context` / `--max-gen` | 指定语言（默认自动）/ system 上下文 / 生成上限 |
| `--prefill-batch N` | 分批 prefill 批大小；校验批数与统计输出（S=40/bs=8 → 5 批） |
| `--cancel-after N` | 流式回调演示：第 N 个正文增量回调后取消，校验 `cancelled` + `stage` + 部分正文前缀 |
| `--window-ref dir` | 跨窗口 golden 对齐模式（见 §5.3），需与转换模型一致的 `--gguf` 或 config 内 model_path |
| `--check-errors` | 演示五类拒绝路径 + 重复 Init 幂等（下） |

错误分类演示（`--check-errors`，全部被显式分类拒绝）：

```text
[empty-input      ] rejected(bad_input      ): empty input: n_samples=0
[wrong-rate       ] rejected(bad_input      ): unsupported sample_rate=44100 (this engine only accepts 16000 Hz; ...)
[too-long         ] rejected(audio_too_long ): audio too long: n_samples=... (...) > limit 30s
[nan-input        ] rejected(bad_input      ): pcm contains NaN/Inf at sample 15200
[context-insuff   ] rejected(context_insufficient): context insufficient: prefill 163 tokens + 1 gen > n_ctx 64
[re-init          ] idempotent OK
```

## 4. 在 C++ 中接入

### 4.1 阻塞模式（v0.3 用法不变）

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

const qwen3_asr_response r = engine.Transcribe(req);   // cb 省略 = 阻塞
if (!r.ok) fprintf(stderr, "error(%s): %s\n", qwen3_asr_error_str(r.err), r.error.c_str());
else       printf("[%s] %s\n", r.language.c_str(), r.text.c_str());
```

### 4.2 文本流式 + 取消

```cpp
std::atomic<bool> cancel{false};
qwen3_asr_request req = /* 同上 */;
req.cancel = &cancel;            // 其他线程随时置位取消

const qwen3_asr_response r = engine.Transcribe(req,
    [](qwen3_asr_stream_event ev, const std::string & piece) -> bool {
        if (ev == QWEN3_ASR_STREAM_LANG) { /* 协议头：如 "Chinese"，无语音时 "None" */ }
        else                            { /* 正文增量：完整 UTF-8，可直接渲染 */ }
        return true;                    // 返回 false = 调用方取消（等价置位 cancel）
    });

if (r.finish_reason == QWEN3_ASR_FINISH_CANCELLED)
    printf("cancelled at stage=%s, text so far=%s\n",
           qwen3_asr_stage_str(r.stage), r.text.c_str());
```

### 4.3 接口约定

- **线程模型**：`Init` / `Transcribe` 互斥串行，并发调用排队逐个执行；重复 `Init` 幂等。回调只在推理线程内同步调用，**回调内不得再进入本引擎任何接口**（重入即死锁）；跨线程投递到 UI 队列是调用方责任。
- **请求字段**：`pcm` / `n_samples` / `sample_rate` / `context` / `language` / `max_gen_tokens`（≤0 用默认）/ `cancel`（可空）/ `prefill_batch_size`（0 用默认）/ `want_debug`。
- **响应字段**：`ok` / `err`（枚举）/ `error` / `text` / `language` / `finish_reason`（含 `cancelled`）/ `stage` / `stats`；`want_debug=true` 时另有 `T` / `A` / `S` / `n_gen` / `gen_ids` / `raw` / `mel` / `log_mel_max`。
- **校验先于计算**：全部输入校验（空 / 采样率 / 超长 / NaN / 参数非法 / 容量）在任何计算前完成，失败无副作用。
- **取消粒度**：检查点在音频 CNN 分块批、prefill 批、decode step 边界；单个大算子完成前无法中断。
- **每请求独立**：入口重置 KV；同一输入重复调用结果逐 token 一致。

引擎配置字段（`asr_conf_v04.json`）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `model_path` | 必填 | `qwen3-asr` 架构 GGUF 路径 |
| `n_ctx` | `0` | KV 容量；0 = `min(4096, n_ctx_train)` |
| `max_audio_seconds` | `30` | 单请求音频时长上限（秒）；更长仍受 `n_ctx` 容量限制 |
| `max_gen_tokens` | `64` | 请求未指定生成上限时的默认值 |
| `prefill_batch_size` | `128` | 混合 Embedding 分批 prefill 批大小；`<=0` = 单批 |

## 5. 验证：golden 对齐

验证需要 Python（`conda activate cv`）；产品推理链路本身不需要。

### 5.1 短音频 golden（v0.2/v0.3 基线，回归保留）

```bash
python scripts/asr/export_asr_reference.py "$MODEL_DIR" \
    --wav data/test_asr_001.wav --out-dir work/asr_ref_v02 --max-new-tokens 128
```

生成 `auto/`（自动语言）与 `lang/`（预置 `language Chinese<asr_text>`）两套案例。短音频回归运行 v0.3 示例或 v0.4 示例皆可（v0.4 的 `--ref-dir` 校验阻塞结果 gen_ids 与 golden `decode_ids.npy` 逐 token 一致，含 `--prefill-batch` 分批；流式结果再与阻塞结果逐字节比对，间接覆盖两种模式）：

```bash
./build/examples/asr/api_test/asr_transcribe_v04 \
    --config build/examples/asr/api_test/asr_conf_v04.json \
    --ref-dir work/asr_ref_v02/auto --prefill-batch 8
```

### 5.2 短音频判定口径（同 v0.3）

| 检查 | 阈值 | 实测基线（data/test_asr_001.wav，T=190） |
| --- | --- | --- |
| Log-Mel vs golden `mel.npy`（v0.3 示例） | `max_abs < 1e-4` 且 `rmse < 1e-5` | max_abs=2.13e-5，RMSE=6.17e-7 |
| 生成 token vs golden `decode_ids.npy` | 数量与逐 token 完全一致 | auto 8/8，lang 5/5（V0.4 全部通过，含 bs=8 分批） |
| 阻塞 vs 流式 | 最终 text 逐字节一致 | 一致（24 字节） |

### 5.3 跨窗口 golden（v0.4 新增）

参考语义以官方 `get_audio_cu_seqlens`（varlen 分块、窗口内双向/窗口间隔离；window=104 位置）为准，用显式块对角 mask 的 eager SDPA 复现并导出：

```bash
# 导出 20s 合成音频的跨窗口参考（T=2000，A=260，窗口边界 [0,104,208,260]）
python scripts/asr/export_asr_window_reference.py "$MODEL_DIR" \
    --out-dir work/asr_ref_win --seconds 20

# C++ 音频塔逐段对齐（window_bounds / conv_out / 逐层 hidden / features）
./build/examples/asr/api_test/asr_transcribe_v04 \
    --config build/examples/asr/api_test/asr_conf_v04.json \
    --window-ref work/asr_ref_win/cross
```

判定口径（容差沿用 v0.2 的 `kAudioRmseTol=0.20`，f16 权重 vs fp32 参考）：

| 检查 | 实测基线（20s，A=260，3 窗口） |
| --- | --- |
| 窗口边界 vs `window_bounds.npy` | [0,104,208,260] 逐值一致 |
| conv_out / 逐层 hidden / features 的 RMSE | 2.7e-4 / 8.1e-4 / 6.2e-5 |

30 秒端到端实测：T=3000、A=390（4 窗口）、S=405 入 KV（n_ctx=4096）；重复调用一致、阻塞与流式一致、RTF≈0.70。

### 5.4 v0.2 逐段数值定位（可选构建）

v0.2 六段对齐工具（conv_out / 逐层 / 投影 / ids / 融合 / logits / teacher forcing）默认不构建：在 [examples/asr/api_test/CMakeLists.txt](examples/asr/api_test/CMakeLists.txt) 打开 `asr_align_v02` 的 `add_executable` 重新构建后：

```bash
./build/examples/asr/api_test/asr_align_v02 \
    --gguf models/qwen3-asr-0.6b-f16.gguf \
    --ref-dir work/asr_ref_v02/auto --mode auto --max-gen 160
```

用法细节（`--mode` 须与案例一致、`--max-gen` 应覆盖参考 `n_gen`）见 [v0.2 代码分析](doc/asr/code_analysis_asr-v0.2.md)与工具内 `--help`。

### 5.5 LLM 回归（共享 Decoder 计算未被 ASR 改动破坏）

LLM 四类调用示例默认不构建：在根 [CMakeLists.txt](CMakeLists.txt) 打开 `add_subdirectory(examples/llm/api_test)` 后重新构建：

```bash
./build/examples/llm/api_test/infer_nothink_blocking --config build/examples/llm/api_test/llm_infer_conf.json
# 另三类：infer_nothink_stream / infer_yesthink_blocking / infer_yesthink_stream
```

v0.4 验收时四类示例（各 3 用例）通过，正常输出文本 / 工具调用（STOP / TOOL_CALLS）。

## 6. 排查问题

| 现象 | 优先检查 |
| --- | --- |
| Python 提示缺少模块（转换 / 导出参考时） | 是否 `conda activate cv`；推理链路与 ASR 示例无 Python 依赖 |
| 找不到模型文件、架构或张量数不符 | `MODEL_DIR` 是否指向原始完整模型；GGUF 是否由本工程 ASR 转换脚本生成（`general.architecture = qwen3-asr`） |
| WAV 被拒绝（channels / 位深 / 采样率） | 输入需转成单声道 16 kHz 16-bit PCM WAV 后再试；本版不自动重采样或混音 |
| 引擎报 `audio too long ... > limit 30s` | 超过配置上限（默认 30 秒）；按需提高 `max_audio_seconds`，同时确认 `n_ctx` 容量足够 |
| 引擎报 `context insufficient: prefill ... > n_ctx` | KV 放不下（prefill+生成）；提高 `n_ctx`（受 `n_ctx_train` 截断）或缩小音频 |
| 流式回调从未触发 `STREAM_TEXT` | 无声 / 纯静音音频可能只产生 `language None` 协议头；是否把语言协议 token 误当转写内容（§3 字段表） |
| `--cancel-after 1` 取消后 `stage=prefill` | 正常语义：首 token 在 prefill 末位产出，首个 TEXT 回调即取消时尚未进入 decode step；`--cancel-after 2` 则为 `stage=decode` |
| 取消后回调看起来"迟到了一点" | 取消粒度限制：不打断单个 ggml 算子，当前批 / step 完成后才生效（设计文档 §5.2 第 2 条） |
| `[window_bounds]` / golden FAIL | `--window-ref` 是否使用同一原始模型导出的参考；窗口参考须用 `export_asr_window_reference.py` 生成 |
| `[gen_ids] FAIL` 但 `[log_mel]` OK | 生成上限是否覆盖参考 `n_gen`、`--language` 是否与参考一致；再跑 asr_align_v02 定位音频塔 / 融合段 |
| Log-Mel 对齐但文字与录音不符 | 数值对齐与识别质量是两件事；检查音频内容与音质，准确率回归属 v1.0 |
| `finish=length` / `context_full` / `cancelled` | 均非错误：生成上限 / KV 耗尽 / 调用方取消的结构化语义；按 `stage` + `error`（仅失败）诊断 |

## 7. 后续版本

持续音频输入与可修订的真正流式转写（Start/Push/Finish、partial/final 协议、缓存失效）属 `asr/v0.5`；API/格式冻结、质量与资源回归、发布契约属 `asr/v1.0`。GPU、量化、VAD、重采样等按需求另开专题。明细见 [版本规划](doc/asr/design_asr.md)。
