# Qwen3-ASR-0.6B 推理版本规划（讨论稿）

## 一、版本规划总览

### 1.1 规划目标与开发基线

基于已实现的 `llm/v1.0`，逐版本构建 Qwen3-ASR-0.6B 的 **GGUF + ggml / C++ 推理能力**，从 Decoder 验证推进到离线转写、音频流式及稳定发布。

- 代码基线：`llm/v1.0` 分支的 `68f428ebcfaa4cd11229b75b697ccdc703ae112f`。核对时未发现本地 tag，使用 commit 固定起点。
- 目标模型：[ModelScope Qwen/Qwen3-ASR-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B)，使用原始版本，不混用 `-hf` 权重、配置和实现。
- 已确认方向：CPU 离线优先；LLM 与 ASR 共存并共享底层 Decoder。
- 首阶段：v0.1～v0.3，交付可验证的纯 C++ 短音频离线转写。
- 后续阶段：v0.4～v1.0，逐步完善 API、较长音频、流式和发布质量。
- 状态：asr/v0.1 已实施并通过验收（见 §2.6 实施记录）；v0.2 及以后仍为计划，版本名称不表示对应 Git 分支或 tag 已创建。

### 1.2 各版本实现内容与预期效果

| 版本 | 本版新增实现内容 | 完成后能达到的效果 | 阶段定位 |
| --- | --- | --- | --- |
| `asr/v0.1` | 完整 ASR GGUF 转换与加载、外部 Embedding 输入、Decoder 数值对齐 | C++ 能读取 ASR 权重，以参考融合 Embedding 为输入计算 logits 并生成 token | 证明现有 LLM Decoder 可以用于 ASR |
| `asr/v0.2` | ggml 音频 Encoder、ASR Prompt、音频特征替换 | 输入参考 Mel，C++ 完成音频编码和文本解码，得到转写结果 | 打通模型主体计算 |
| `asr/v0.3` | C++ Log-Mel、PCM/WAV 输入、最小阻塞 ASR API | 输入短音频，完全在 C++ 中输出语言与转写文字 | 首个完整离线可用版本 |
| `asr/v0.4` | 完整离线 API、文本流式、取消、统计、分批 prefill、跨窗口验证 | 完整音频可一次返回或逐 token 返回文字，支持经验证的较长输入 | 完善应用接入能力 |
| `asr/v0.5` | 持续 PCM 输入、流式状态、文本回改、结束刷新 | 音频持续输入时返回暂定转写，结束后给出最终结果 | 真正音频流式版本 |
| `asr/v1.0` | API/格式冻结、质量回归、资源测量与针对性优化 | 在明确支持范围内提供可重复验证、可稳定集成的 ASR 推理能力 | 稳定发布版本 |

### 1.3 版本推进关系

```text
llm/v1.0
  → asr/v0.1：参考融合 Embedding → C++ Decoder → token
  → asr/v0.2：参考 Mel → C++ AudioEncoder + Decoder → 转写
  → asr/v0.3：PCM/WAV → C++ 全链路 → 转写
  → asr/v0.4：完整音频 → 阻塞结果 / 文本流式结果
  → asr/v0.5：持续音频 → 可修订的暂定结果 → 最终结果
  → asr/v1.0：固化能力范围、质量与发布契约
```

每个版本以前一版本通过验收为前提。v0.1/v0.2 是可运行的验证里程碑，不宣称支持直接输入音频完成纯 C++ 转写；v0.4 的文本流式不等同于 v0.5 的音频流式。

## 二、asr/v0.1 版本规划：GGUF 转换与 Decoder 复用

### 2.1 版本目标

证明 ASR 文本子模型可以复用当前 Qwen3 Decoder 计算逻辑，建立完整权重加载和数值验证基础。

### 2.2 要实现的内容

1. **固定模型和参考环境**：记录下载来源、revision、文件哈希、官方源码/依赖版本、dtype 和 attention backend。ModelScope revision 不套用 HF SHA。
2. **建立参考数据**：保存短音频对应的 PCM、Prompt、IDs、Mel、融合 Embedding、logits 和生成 token，自动语言与指定语言分别准备。
3. **新增 ASR 转换入口**：复用 GGUF writer 和文本映射设施，单文件保存完整音频塔、文本 Decoder、Embedding、LM Head、Tokenizer 和必要配置，不改变现有 LLM 转换行为。
4. **定义并校验 GGUF 契约**：包含音频前处理、分块参数、音频 token ID 和 EOS 集合；loader 校验架构、格式版本、参数类型、必需张量和维度。自定义名称在本版编码前完成字段级评审，不冒充已有通用标准。
5. **检查全部权重去向**：以文档基线 612 个存储张量建立映射清单；实际下载不同时先核对 revision。保留 Embedding 与 LM Head 两份矩阵，不按 `tie_word_embeddings` 自动去重。
6. **确定浮点路径**：矩阵优先 F16，归一化/bias 等按算子要求保留 F32，提供 F32 调试导出能力，暂不量化。
7. **扩展 Decoder 输入**：支持 token IDs 或外部 `[1024,S]` Embedding，共享层计算；只对最后位置执行 LM Head，同步调整 logits 读取偏移。
8. **建立对齐工具**：比较逐层 hidden state、prefill logits、top-k 和 teacher forcing 下的多步 decode。

### 2.3 完成后的效果

- 能将目标 ASR 模型转换为本工程可加载的完整 GGUF。
- 能将 Python 导出的融合 Embedding 输入 C++ Decoder，得到可与官方参考比较的 logits 和 token。
- 能区分模型转换、Decoder 计算和后续音频处理的问题，不必等端到端完成才定位误差。
- 原有 LLM token ID 输入路径继续可用。

### 2.4 交付物与验收标准

- 交付 ASR 转换入口、GGUF 参数/张量映射清单、加载逻辑、Decoder 双入口和数值对齐示例。
- 所有存储权重去向明确；缺失张量、错误架构和不支持的格式版本可诊断。
- 保存逐层误差及 logits/top-k 对比报告；容差根据参考精度和实测建立，不要求不同 dtype bitwise 一致。
- 现有 LLM 四类 API 示例回归通过。

### 2.5 本版不实现

不实现 C++ 音频 Encoder、Log-Mel、WAV 转写或流式 API。参考中间数据是验证输入，不是最终部署方案。

### 2.6 实施记录（2026-09-19，已验收）

#### 交付物

| 交付物 | 位置 | 说明 |
| --- | --- | --- |
| 转换入口 | `tools/convert_asr_hf_to_gguf.py` | 文本/音频映射表、F32 保留规则、tokenizer 构建、契约键写入、612 张量映射清单输出 |
| ASR GGUF | `models/qwen3-asr-0.6b-f16.gguf` | 1795.3 MB；612 张量全部映射（text=311，audio=301），与文档基线一致 |
| 张量映射清单 | `doc/asr/asr_v01_tensor_mapping.json` | 每个存储张量的 HF 名 → GGUF 名 → 去向，无静默丢弃 |
| 加载扩展 | `src/model.h` / `src/model.cpp` | `qwen3-asr` 架构；`asr.*` 元数据读取；音频塔 301 张量加载与形状校验 |
| 计算图扩展 | `src/graph.h` / `src/graph.cpp` | 外部 Embedding 入口、仅末位 LM Head、逐层输出导出（`layer_out_{i}`） |
| 参考基线导出 | `tools/export_asr_reference.py` → `work/asr_ref/{auto,lang}/` | 合成 5s 音频、官方 Mel、fp32 音频塔逐行复现、transformers Decoder 基线（见 §2.2 第 1/2 条环境记录） |
| 数值对齐示例 | `examples/asr/api_test/asr_align_v01.cpp`（挂 `FROSTFALL_BUILD_EXAMPLES`） | 逐层 hidden / prefill logits / teacher forcing / 自由贪心生成四段对比 |
| 对齐报告 | `doc/asr/asr_v01_align_report_{auto,lang}.log` | 实测误差全文 |

#### GGUF 契约落地（对应 §2.2 第 3/4/5/6 条）

- `general.architecture = "qwen3-asr"`；文本超参沿用 `qwen3.*` 键；新增 `asr.*`（audio token id、EOS 集合数组、chat_template、support_languages）、`asr.audio.*`（音频塔超参）、`asr.mel.*`（Log-Mel 前处理契约，供 v0.3 C++ 实现对齐）。
- tokenizer：151,936 = 151,705 真实 token（151,643 基础 + 62 added）+ 231 dummy 补齐；special → CONTROL，非 special added（如 `<asr_text>`）→ USER_DEFINED。
- 张量布局：conv 权重 PyTorch `[out,in,KH,KW]` → ne={KW,KH,in,out}，im2col 兼容无物理重排（v0.2 使用）；矩阵 F16；全部 bias 与 RMSNorm/LayerNorm 权重 F32（按 `*.bias` / `*norm.weight` 后缀规则 + `ln_post.weight` 显式补充；实施中发现裸子路径清单匹配不到带层前缀的完整名导致 norm 误转 F16，已修正为规则匹配）。
- `embed_tokens` 与 `lm_head` 两份矩阵独立保留，不按 tie 去重。
- CNN 输出通道数（480）不在 config 中，加载时从 `conv1.weight` 的 ne[3] 推导并自洽校验（`asr.audio.conv_channels`）。

#### 关键实现决策

- **仅末位 LM Head**（`logits_last_only`，默认开）：prefill 的 LM Head 从 `[n_vocab,S]` 缩为 `[n_vocab,1]`。LLM 引擎只读末位 logits，行为不变（已回归验证）。
- **embd 双入口**：调用方在 no-alloc ctx 中创建 `[n_embd,n_tokens]` F32 输入张量（已 `set_input`），经 `qwen3_graph_params.embd` 传入，作为第 0 层输入；与 token 入口互斥。
- **音频塔**：v0.1 仅加载与形状校验，不参与计算（计算属 v0.2）。
- **参考精度**：fp32（高于官方 bf16 推理），Decoder 侧 `tie_word_embeddings=False` 独立加载两份矩阵。

#### 对齐结论（两案例均 PASS）

参考 fp32 vs 本工程 F16 权重 + F16 KV cache，容差按实测建立：

| 指标 | auto（S=80，n_gen=4） | lang（S=83，n_gen=3） | 验收标准 |
| --- | --- | --- | --- |
| 逐层 hidden（排除 sink 的 rmse） | ≤ 0.017 | ≤ 0.017 | ≤ 0.05 |
| prefill 末位 logits max_abs / top10 | 0.055 / 10·10 | 0.022 / 10·10 | ≤ 0.5 且 ≥ 9·10 |
| teacher forcing decode max_abs / top10 | 0.032 / 10·10 | 0.019 / 10·10 | ≤ 0.5 且 ≥ 9·10 |
| 自由贪心生成 | 4·4 token 一致 | 3·3 token 一致 | 与参考完全一致 |

- **attention sink 现象**：`<|im_start|>`（token 0）的 hidden 达 10^3 量级，F16 KV 量化误差在该位置逐层放大，深层起波及其他 sink 型 token；绝对偏差大（最大约 400）但相对误差约 0.15%。末位 logits 与生成结果不受影响。因此 hidden 判定采用排除 token 0 的 rmse；末层 hidden 不单独判定（其唯一消费者是末位 logits，由 logits 判定覆盖）。
- **LLM 回归**：四类示例（nothink/yesthink × stream/blocking，各 3 测试）全部通过。

#### 与规划的差异与边界

- Mel 由官方 WhisperFeatureExtractor（Python）提取，C++ Log-Mel 属 v0.3；音频塔 C++ 计算属 v0.2。
- 测试音频为合成信号（非真实语音），auto 案例参考生成 `language None<asr_text>`、lang 案例生成 `哦。`——v0.1 验收目标是数值对齐，不是识别质量。
- 其余按 §2.5 边界：不含 WAV 输入、流式 API、量化。

## 三、asr/v0.2 版本规划：音频 Encoder 与模型主体闭环

### 3.1 版本目标

在 v0.1 的基础上补齐音频模型计算，做到“参考 Mel 输入，C++ 输出转写”。

### 3.2 要实现的内容

1. **音频 ggml 计算图**：三层 Conv2d + GELU、channel/frequency 展平、conv_out、局部正弦位置编码、18 层音频 Transformer、LayerNorm 与输出投影。
2. **数值和布局对齐**：验证卷积类型组合、bias、LayerNorm、GELU 形式、张量布局、尾块 padding 和展平顺序。
3. **分块和有效长度**：CNN 每 100 个有效 Mel 帧分块，音频位置数采用 `A(T)=13×floor(T/100)+ceil((T mod 100)/8)`，不能使用整段 `ceil(T/8)`。
4. **ASR Prompt 与 Tokenizer**：加载 ASR 词表和 Added Tokens，构造自动/指定语言模板，按音频位置数扩展占位符，不添加 Thinking 前缀。
5. **音频特征融合**：只替换 `audio_pad` 位置，保留音频起止标记的文本 Embedding；校验特征数量、顺序与维度。
6. **接通生成与解析**：混合 Embedding prefill 后使用 token ID 增量 decode，输出原始 token、语言与转写文字，便于逐阶段比较。

### 3.3 完成后的效果

- Python 只负责提供 Mel 和参考数据，音频编码、特征融合、文本 Decoder 与生成在 C++ 中完成。
- 输入一段短音频对应的 Mel，可得到转写结果。
- 可以分别检查 CNN、音频 Transformer、投影和融合输入误差，确认整个模型主体计算正确。

### 3.4 交付物与验收标准

- 交付音频 Encoder、ASR Prompt/融合/解析模块，以及参考 Mel 驱动的推理示例。
- 对比 CNN 各层、音频各层、投影、融合 Embedding 和 Decoder 输出。
- 覆盖 99/100/101、199/200/201 个有效 Mel 帧及尾块 padding。
- 首轮使用单注意力窗口内短音频，明确参考 backend；保留 v0.1 与 LLM 回归。

### 3.5 本版不实现

不实现 C++ 音频前处理和直接 WAV 输入；不宣称支持跨窗口长音频或真正音频流式。

## 四、asr/v0.3 版本规划：纯 C++ 短音频离线转写

### 4.1 版本目标

补齐音频输入与 Log-Mel，交付首个无需 Python 运行时的完整离线转写版本。

### 4.2 要实现的内容

1. **PCM/WAV 输入**：核心接口接收单声道 16 kHz float32 PCM；示例支持 PCM16 WAV，采用 `/32768` 转换。其他采样率和格式明确拒绝，不只修改采样率标签。
2. **C++ Log-Mel**：400 点 FFT、160 点 hop、周期 Hann、reflect padding、128 维 Slaney Mel、log10 压缩与官方归一化；对齐最后帧删除和有效长度。
3. **最小阻塞接口**：提供独立 `AsrEngine` 的 `Init`、`Transcribe`；请求包含 PCM、采样率、context、language 和生成上限，响应包含 text、language、完成原因和调试输出能力。
4. **离线编排**：每请求重置 Decoder KV；一次音频编码后完成混合 prefill 和增量 decode，不在每个生成 step 重跑音频塔。
5. **生成约束**：默认贪心、关闭重复惩罚，正确处理 EOS 集合、`<asr_text>`、指定语言前缀、长度截断和上下文不足。
6. **输入边界**：初始公开范围建议为单请求、单音频、不超过 8 秒；校验空输入、NaN/Inf 和容量，超限报错，不静默截断。这是工程边界，不是模型能力上限。

### 4.3 完成后的效果

- 用户可以输入一条支持格式的短音频，直接得到语言和转写文字。
- 支持自动语言、指定语言和 context；整个推理链路不再依赖 Python。
- 能明确区分识别完成、生成截断和输入错误。
- 同一工程继续提供原有 LLM 能力，ASR 与聊天接口相互独立。

### 4.4 交付物与验收标准

- 交付 C++ 音频前端、最小 ASR API、配置样例及 `examples/asr/api_test/` 离线示例。
- 对齐 PCM → Mel → 音频特征 → 融合输入 → token → 文本全链路。
- 覆盖短中文、英文、数字、标点、静音，以及自动/指定语言、空/非空 context。
- 重复调用不串 KV，错误和截断状态明确；LLM 与前序数值测试回归通过。

### 4.5 本版不实现

不承诺长音频、文本流式、麦克风流式、GPU、量化、重采样、多种压缩格式或时间戳。

## 五、asr/v0.4 版本规划：离线 API、文本流式与较长音频

### 5.1 版本目标

将最小离线闭环完善为方便应用集成的接口，并扩展经验证的输入长度。

### 5.2 要实现的内容

1. **完整初始化与错误协议**：结构体/JSON 初始化、串行互斥、重复调用、错误分类和回调重入约束。
2. **取消机制**：在音频块、prefill 批次、decode step 边界检查取消，返回阶段状态与部分结果；不承诺立即打断单个 ggml 算子。
3. **文本流式**：完整音频输入后逐 token 回调转写内容，缓冲不完整 UTF-8 和语言协议，不使用 Thinking/ToolCall splitter。
4. **分批 prefill**：对混合 Embedding 分批，确保切片、位置、mask 和 `n_past` 同步，降低峰值内存。
5. **跨窗口与较长音频**：固定官方参考后端并验证块对角双向注意力，扩展至 30 秒及上下文容量内的更长音频，保留显式时长和容量校验。
6. **性能统计**：前处理、Encoder、prefill、decode、首模型 token、首可见文字、RTF 和峰值内存。

### 5.3 完成后的效果

- 应用可选择一次取得最终转写，或在完整录音处理后逐 token 展示文字。
- 可以取消请求，读取处理阶段和耗时，明确获知截断或错误。
- 可以处理已经跨窗口验证的较长输入，不再局限于 v0.3 的短音频范围。
- 分批 prefill 减少一次性处理大量位置带来的内存压力，收益通过实测确认。

### 5.4 交付物与验收标准

- 交付阻塞/文本流式 API 示例、取消与错误测试、跨窗口测试和性能统计报告。
- 同一输入在阻塞与文本流式模式下最终结果一致。
- 验证单窗口、跨窗口、多窗口、上下文不足、取消及重复调用。
- 测量首可见文字延迟与 RTF，不将首个语言协议 token 当成首个转写文字。

### 5.5 本版不实现

不是边录边识别；不直接承诺 20 分钟输入、无界长音频、实时性能或永久增量音频缓存。

## 六、asr/v0.5 版本规划：持续音频输入与真正流式转写

### 6.1 版本目标

在离线链路稳定后，支持音频持续输入与可修订的中间转写。

### 6.2 要实现的内容

1. **独立流状态**：定义 Start/Push/Finish/Reset 生命周期，管理累计 PCM、历史转写和处理进度。
2. **参考流式策略**：首轮遵循固定官方参考的累计音频、历史文本前缀回退和尾块刷新行为；输入触发节奏在本阶段评审时确定。
3. **结果协议**：区分 partial/final，返回当前完整文本和版本号，允许修改尾部，不能伪装为只增不减的 delta。
4. **缓存有效性**：新音频可能改变窗口特征及转写前的序列位置，明确重算和失效规则，不直接在旧文本 KV 后追加新音频。
5. **结束与异常处理**：尾音频刷新、暂停、取消、重复 finish、reset 后重用以及无效调用顺序。
6. **流式测量**：基于目标硬件测量首字延迟、更新间隔、回改范围和 RTF；指标目标进入本阶段前另行确认。

### 6.3 完成后的效果

- 应用持续提交 PCM 时，可在录音结束前显示暂定转写。
- 后续音频到来时，可以修订先前暂定结果，而不是重复追加错误文字。
- 输入结束后会处理残余音频，并明确返回最终结果。
- 应用能区分“仍可能变化的文字”和“本次已完成的结果”。

### 6.4 交付物与验收标准

- 交付流式状态/API、分块 PCM 输入示例、partial/final 协议及状态转换测试。
- 验证尾块、暂停、取消、重复 finish、文本回改和缓存失效。
- 比较流式最终结果与离线结果，记录差异及质量指标，不强求两种策略逐 token 完全一致。
- 提供目标硬件上的时延和资源报告，不以“接口流式”代替实时性能验收。

### 6.5 本版不实现

不默认加入麦克风采集驱动、VAD、说话人分离、时间戳；不承诺音频塔和 Decoder 全部严格增量且永不重算。

## 七、asr/v1.0 版本规划：稳定发布

### 7.1 版本目标

将已验收的 ASR 能力固化为可重复构建、验证和集成的稳定版本。

### 7.2 要实现的内容

1. **冻结公开契约**：API、配置字段、GGUF schema、支持输入范围、完成/错误/取消/流式语义和兼容政策。
2. **完善质量回归**：固定模型和测试集版本，建立 CER/WER、边界输入、重复调用及 LLM 共存回归。
3. **资源基线与优化**：测量各阶段耗时、RTF、首字延迟和峰值内存，按实测瓶颈优化并检查质量回退。
4. **发布材料**：构建方法、模型转换方法、示例、已知限制、参考环境和性能复现步骤。
5. **一致性检查**：文档声明、示例行为和实际支持能力一致，不将路线图内容写成已实现能力。

### 7.3 完成后的效果

- 使用者能在明确的输入与硬件范围内稳定集成离线/流式能力。
- 模型格式与 API 行为可预期，错误可诊断，质量和资源开销可复现。
- 后续优化或扩展有固定回归基线，避免破坏 ASR 或原有 LLM 能力。

### 7.4 交付物与验收标准

- 交付稳定 API/格式说明、完整示例、回归集及准确率/性能/内存报告。
- 所有公开声明能力通过对应测试；测试环境、输入范围、计时边界和已知限制明确。
- 性能目标按已确认硬件与需求验收，不引用官方不同后端的最优数字作为本工程承诺。

### 7.5 不作为默认发布前提的能力

GPU、量化、batch、VAD、重采样、额外音频格式和时间戳/ForcedAligner 按需求另开专题，不要求全部完成才发布 v1.0。

## 附录 A：各版本共用的实现约束

- 保留 `Frostfall::InferenceEngine`；ASR 使用独立请求响应，不把音频放入 `Message.content`。
- 复用 `src/model.*` 的 GGUF/backend 加载设施、`src/graph.*` 的 Decoder、`src/kv_cache.*`、BPE 和贪心采样基础；首阶段不做全工程目录迁移。
- ASR 使用自己的 Decoder 权重和词表，复用计算代码不等于复用聊天模型权重。
- 音频向量替换占位 Embedding 后进入因果 Decoder，不新增 Cross-Attention；音频 Encoder 自己使用非因果注意力。
- 原始 ASR 三轴位置相同的路径可复用 NEOX RoPE，须验证；不推广到其他多模态位置编码。
- 示例位于 `examples/asr/api_test/`，沿用 `FROSTFALL_BUILD_EXAMPLES`；LLM 示例保留回归。
- 公共验证资产包括 PCM、Mel、Prompt 字节、IDs、音频特征、融合 Embedding、logits 和原始输出。数值误差记录最大绝对误差、RMSE、必要时余弦相似度和 top-k。
- teacher forcing 用于定位计算误差，端到端另外评估文本质量，不用后处理掩盖 logits 偏差。
- 模型集成测试和无需权重的单元测试分离；每个版本保留前序已交付能力的回归。

## 附录 B：风险与评审事项

- 当前没有实际 ASR GGUF 运行结果，不承诺 CPU 实时或准确率指标。
- 固定官方源码可能存在跨窗口 attention 后端差异，v0.4 跨窗口验收前必须明确参考语义。
- 完整 F16 权重约 1.75 GiB，4096 位置 F16 KV 约 448 MiB，另有音频激活和前处理开销；以实际测量为准。
- `RTF=处理耗时/音频时长`，RTF 小于 1 不等于低首字延迟；报告须注明计时是否包含录音等待。
- 时间戳需要额外对齐模型，不能按音频位置乘固定毫秒数直接推导文字时间戳。
- 自定义 GGUF 字段、具体 API 签名和后续性能目标在对应版本实施前评审。本文仍为讨论稿，不表示所有版本已获实现授权。

## 附录 C：参考文档

- [模型资源与开发基线](../qwen_model/Qwen3-ASR-0.6B_Resource.md)
- [模型架构与配置](../qwen_model/Qwen3-ASR-0.6B_Model_Architecture_And_Configs.md)
- [端到端推理与 llm v1.0 适配](../qwen_model/Qwen3-ASR-0.6B_Model_Architecture_And_Inference.md)
- [LLM v1.0 设计](../llm/design_llm_v1.0.md)
