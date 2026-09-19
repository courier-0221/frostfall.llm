# Qwen3-ASR-0.6B 端到端推理与 llm v1.0 适配

## 一、目标与阅读前提

本文回答：一段录有“今天天气怎么样”的音频，如何变成文字？其中哪些可以复用当前 frostfall.llm v1.0，哪些必须新增？

模型资源、版本和未验证事项见 [资源文档](Qwen3-ASR-0.6B_Resource.md)，各层尺寸见 [架构与配置](Qwen3-ASR-0.6B_Model_Architecture_And_Configs.md)。本文仅分析并提出开发建议，不代表下面的 ASR 接口和模块已经实现。

核心区别：LLM 收到“今天天气怎么样”可能回答天气问题；ASR 收到这句话的语音，应输出转写文字，而不是回答问题。要做语音助手，链路应为 `音频 → ASR 文字 → 现有 LLM → 回答/Tool Call`。

## 二、端到端数据流

```text
音频文件 / PCM
  │
  ├─ 解码容器、单声道、重采样到 16 kHz、float32
  │
  ├─ Whisper 风格 Log-Mel 特征提取
  │      └─ input_features + feature_attention_mask
  │
  ├─ 音频编码：CNN → 音频 Transformer → 投影
  │      └─ audio_features [A,1024]
  │
  └─ ASR 模板 + context + 可选 language
         └─ 将一个 audio_pad 扩展为 A 个占位 token
                └─ BPE → input_ids [S]
                       └─ 查表并替换音频位置 → inputs_embeds [S,1024]
                              └─ Decoder prefill → 各层 KV + 首个 logits
                                     └─ 贪心 decode → token → 文本
                                            └─ ASR 协议解析
                                                   └─ language + text
```

形状记号：B 为请求 batch，T 为有效 Mel 帧数，A 为音频编码后位置数，S 为 prefill 的混合序列长度。以下先采用 `B=1`、完整音频离线转写。

## 三、阶段 1：音频输入标准化

官方 `normalize_audio_input` 支持本地路径、URL、Base64 和 `(np.ndarray, sample_rate)`；底层模型真正接收的是特征张量，不是这些输入形式。

标准化步骤：

1. 解码得到波形，保留原采样率。
2. 多声道取均值，得到单声道。
3. 原采样率不是 16 kHz 时实际重采样，不能只改采样率标签。
4. 转为 float32；官方对峰值超过 1 的输入做保守幅度归一化，并裁剪到 `[-1,1]`。
5. 校验空输入、异常值和时长，再提取特征。

首版 C++ 建议限定为单声道 16 kHz PCM/WAV；其他格式通过独立解码模块转换。明确 `int16 → float32 / 32768` 的输入契约，避免把整数幅度直接当作浮点音频。若与官方数组接口做严格比较，还需对齐其幅度处理策略。

第一版不需要把 VAD、降噪、回声消除、说话人分离一起加入模型图；这些是可选前置能力，启用后应重新评估识别率。

## 四、阶段 2：Log-Mel 特征

参考 Transformers 4.57.6 的 [WhisperFeatureExtractor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/feature_extraction_whisper.py)：

```text
16 kHz waveform
 → Hann window，400 点 FFT，160 点 hop
 → STFT 功率谱 |X|²，单边频谱 201 bins
 → 128 维 Slaney Mel 滤波，0～8000 Hz，Slaney normalization
 → log10(max(mel, 1e-10))
 → max(log_mel, 当前样本 log_mel 最大值 - 8)
 → (log_mel + 4) / 4
```

实现对齐注意：

- PyTorch 路径使用 `torch.stft` 默认居中与 reflect padding；默认 Hann 为周期窗。
- 删除 STFT 的最后一个时间帧，再使用对应 Mel 结果；边界处理和有效长度必须一致。
- 不要擅自增加预加重、逐帧均值方差归一化或随机 dither。
- “先按整段音频提取 Mel，再在 Mel 空间切 CNN 块”与“每秒独立算 STFT”在边界和最大值压缩上不等价。
- Processor 设置 `truncation=False`，不强制截成 30 秒。
- 有效帧数来自 `feature_attention_mask`；不能用含 padding 的张量长度代替。

结果：

```text
input_features         [1,128,T_pad]
feature_attention_mask [1,T_pad]
T = feature_attention_mask.sum()
```

对于整秒、无额外 padding 的标准音频，T 约为 `100 × 秒数`。非 hop 整数倍采样长度、极短音频和 batch padding 要直接对齐官方输出形状与 mask，不能只用时长估算。

## 五、阶段 3：音频编码

以 3 秒、有效 Mel 帧数恰为 300 的示意输入为例：

| 模块 | 输入 → 输出，省略请求 batch |
| --- | --- |
| 按 100 帧分块 | `[128,300] → 3 × [128,100]` |
| CNN + GELU × 3 | `[3,1,128,100] → [3,480,16,13]` |
| 展平和 `conv_out` | `[3,13,7680] → [3,13,896]` |
| 局部正弦位置相加 | 每个块的 13 个位置使用局部 0…12 |
| 去掉 padding 并按时间拼接 | `[39,896]` |
| 音频 Transformer × 18 | `[39,896] → [39,896]` |
| LayerNorm + proj1 + GELU + proj2 | `[39,896] → [39,1024]` |

3 秒只是解释形状的例子，不代表本次实际录制、转写了该语句。

必须遵守的约束：

```text
A = 13 × floor(T / 100) + ceil((T mod 100) / 8)
audio_features.shape == [A,1024]
```

不能用一个音频均值向量代表全部声音，也不能将音频 tower 的输出 token 当作离散词表 ID。

音频 Transformer 是非因果的。官方构造了推理窗口边界；不同参考后端在本文固定源码版本中存在 mask 传递差异，详见架构文档第五节。首轮对齐建议使用单注意力窗口内的短音频，之后单独测试跨窗口样本。

## 六、阶段 4：ASR Prompt 与 Tokenizer

### 6.1 自动识别语言

来自 [chat_template.json](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/chat_template.json) 与官方 `_build_text_prompt`，单音频模板为：

```text
<|im_start|>system\n{context}<|im_end|>\n
<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n
<|im_start|>assistant\n
```

上面 `\n` 表示真实换行，不是两个字面字符；代码块分行仅用于阅读，不应额外增加空白。`context` 默认为空，system 消息仍然存在。

### 6.2 强制语言

如果 `language="Chinese"`，在 assistant 前缀后追加：

```text
language Chinese<asr_text>
```

此时语言和正文分隔标记已经是输入的一部分，模型只需生成转写正文。不要把指定语言实现为随意添加“请用中文回答”，也不要自动插入 `<think>\n\n</think>`。

语言 API 使用支持列表中的规范名称，如 `Chinese`、`English`；`zh` 等代码需要上层显式映射，不能直接假设被原始包接受。

### 6.3 占位符扩展

Processor 按 A 将一个 `<|audio_pad|>` 扩展为 A 个：

```text
<|audio_start|><|audio_pad|> × A <|audio_end|>
```

其中 `× A` 只是示意，不是模板字面量。对于上述 3 秒示例，A=39，然后再完成文本分词，得到 `input_ids`。

现有 `src/tokenizer.cpp` 的 byte-level BPE 机制可复用，但必须加载 ASR 的 vocab、merges、Added Tokens。相同 `vocab_size` 并不保证所有 token 的字符串、ID 和 special 属性相同。

首版需要逐字节比较官方 prompt、完整 token IDs、音频起止标记和 39 个占位 ID，尤其注意 `<asr_text>` 是 `special=false` 的 Added Token。

## 七、阶段 5：融合连续音频与文本 Embedding

官方逻辑可以概括为：

```python
inputs_embeds = embed_tokens(input_ids)
mask = input_ids == 151676
inputs_embeds[mask] = audio_features
```

音频开始/结束标记保留其文本 Embedding，只替换 `audio_pad` 位置。这里是替换，不是与占位 Embedding 相加。

必须校验：

```text
count(input_ids == audio_token_id) == A
每个音频块的特征排列顺序 == 对应占位符出现顺序
```

例如把普通文本/控制 token 总数记为 P，则：

```text
S = P + A
inputs_embeds = [S,1024]
```

在 ggml 布局下通常表示为 `[1024,S]`。最小改造可以让现有 Decoder 在“token ID 查表”与“外部 Embedding 输入”两种入口间选择；从进入第 0 层开始的层计算尽量保持一致。

无需为 C++ 实现完整 PyTorch `masked_scatter`：单音频连续区间可用拷贝或拼接完成。但仍要保留清晰的音频位置与长度信息，不能依赖写死的模板偏移。

## 八、阶段 6：Prefill 与自回归 Decode

### 8.1 Prefill

1. 清空本次请求的 Decoder KV 使用长度。
2. 将混合序列送入 28 层 Decoder。
3. 为文本及音频位置统一分配位置编号。无 padding 单样本为 `0…S-1`。
4. 应用文本 Decoder 因果 mask，各层写入全部 S 个位置的 K/V。
5. 对最后一个位置计算 logits，选择第一个输出 token。

音频塔可以先双向处理音频，而 Decoder 仍保持因果注意力；这两种 mask 属于不同网络，不矛盾。

### 8.2 Decode

```text
最后位置 logits → argmax → token y0
Embedding(y0) + position S + 历史 KV → 新 logits → token y1
Embedding(y1) + position S+1 + 历史 KV → 新 logits → token y2
……直到 EOS / 长度限制 / 取消 / 错误
```

官方 `prepare_inputs_for_generation` 在非首步将 `input_features` 设为空，因此同一次离线生成中的音频塔只在 prefill 阶段运行，不应在每个文本 token 生成时重复编码。

官方生成配置为 `do_sample=false`，EOS 集合为 `[151643,151645]`。建议 C++ 基线使用贪心、关闭重复惩罚，不必为了贪心 argmax 先算全词表 Softmax。后续若改变采样策略，应明确视为行为变化。

`<|audio_end|>` 不是生成停止符；`max_new_tokens` 达到上限时，应返回截断状态而不是伪装自然完成。

### 8.3 KV Cache 与计算量

音频特征一旦进入 Decoder，其所有位置都占各层 KV：

```text
F16 KV = n_ctx × 112 KiB
```

30 秒约 390 个音频位置；20 分钟约 15600 个音频位置，仅音频位置的有效 KV 约 `1706.25 MiB`。此外还有模板、context 和转写输出位置。

当前 `src/graph.cpp` 为 prefill 全部位置生成 `[151936,S]` F32 logits。仅 S=390 就约 226 MiB，且还没算模板位置。ASR 长音频前建议优化为只计算最后位置的 LM Head，或分批 prefill 并仅在末批取输出。

分批 prefill 必须保持混合 Embedding 切片、位置和 `n_past` 同步；它不能改变前面音频编码器的窗口定义。

## 九、阶段 7：输出解析

自动语言模式的示意原始输出：

```text
language Chinese<asr_text>今天天气怎么样？<|im_end|>
```

EOS 在生成循环中处理后，解析剩余文本：

```text
language = Chinese
text = 今天天气怎么样？
```

强制语言模式中，模型新生成部分通常只有正文，语言直接沿用请求；不能再要求正文包含 `<asr_text>`。

官方 `parse_asr_output` 还处理：

- 无 `<asr_text>`：将整个结果作为正文，语言为空。
- `language None<asr_text>` 且无正文：返回空语言、空正文。
- 在最终解析前对过度重复字符/模式做启发式修整。

这些是后处理，不是模型权重或采样逻辑。调试应同时保存 raw token IDs、原始文本和后处理结果；不能仅根据修整后的文字判断 C++ logits 是否正确。静音仍可能产生幻觉文字，不能仅靠这一启发式当作可靠 VAD。

输出 UTF-8 可能跨 token 切分。逐 token 回调时应缓冲不完整字节，并缓冲可能被切碎的协议标记。ASR 应使用独立解析器，不经过现有 ThinkSplitter/ToolCallSplitter，以免真实转写中的同名字面量被误解释。

## 十、离线、文本流式和音频流式的区别

| 模式 | 输入何时完整 | 返回内容 | 是否可复用当前 LLM stream 语义 |
| --- | --- | --- | --- |
| 离线转写 | 调用前完整 | 一次性最终文字 | 可复用阻塞编排风格 |
| 完整音频 + 文本 token 流 | 调用前完整 | 逐 token 输出 | 可复用回调基础设施，但换 ASR 解析器 |
| 音频流式转写 | 麦克风持续输入 | 暂定文字会回改，结束时定稿 | 不能只设置现有 `stream=true` |

原始官方包的音频流式算法，见 [qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py)：

1. 默认每累计 2 秒音频触发一次处理。
2. 将新块追加到 `audio_accum`，每次重新提交从开始到当前的累计音频。
3. 前 2 个块不使用历史转写前缀。
4. 后续从上次转写回退末尾 5 个 token，把保留部分追加到 prompt。
5. 生成当前剩余转写，更新完整的 `state.text`；尾部允许被修订。
6. `finish_streaming_transcribe` 刷新不足一块的尾音频。

这是官方上层封装的具体策略，不能理解成“音频 encoder 和 Decoder 的全部 KV 都严格增量、永远不重算”。更不能把上次完整文本 KV 后面直接追加新音频：新音频的序列位置在转写前，且未完成的音频窗口特征可能变化。

该包当前接口限制为 vLLM、单流、无时间戳；这些是本版本工具接口限制，不是对所有未来引擎的限制。

建议 C++ 第一阶段先做离线短音频。后续真正音频流式采用独立状态对象，区分暂定文本和最终文本，并定义替换范围/版本号等协议，避免把回改内容当成只增不减的 delta。

## 十一、长音频与时间戳

这里有四种不同的“分块”，不能混为一个参数：

| 分块层级 | 当前参考行为 |
| --- | --- |
| 音频 CNN 分块 | 每 100 个 Mel 帧，约 1 秒 |
| 音频注意力分块 | 完整块条件下约 104 个编码位置，约 8 秒；注意参考后端差异 |
| 流式输入触发块 | 官方默认 2 秒 |
| 超长音频离线切分 | 普通 ASR 以 1200 秒为目标，在附近寻找低能量边界 |

离线 `split_audio_into_chunks` 的边界搜索允许在目标点附近调整，1200 秒是切分目标，不应把实现描述成每段精确等于或严格不超过 1200 秒。首版若设置硬时长限制，应单独做输入校验。

需要字词时间戳时，先 ASR，再将音频和转写送给 `Qwen3-ForcedAligner-0.6B`。报告描述对齐模型支持至 300 秒，但本文固定 `qwen-asr` 包在 ASR+对齐链路中使用 `MAX_FORCE_ALIGN_INPUT_SECONDS=180` 秒切分目标。模型能力上限与工具策略应分别记录。

不要按 `音频位置索引 × 80 ms` 直接给文字打时间戳：音频位置与文字 token 不是一一对应关系，且 CNN 分块存在取整。

## 十二、基于当前 llm v1.0 的复用边界

| 当前文件/能力 | 可复用部分 | 必须新增或调整 |
| --- | --- | --- |
| `src/model.{h,cpp}` | GGUF 读取、backend 内存加载、文本权重结构 | ASR 类型校验、嵌套参数映射、完整音频塔权重及 bias |
| `src/tokenizer.{h,cpp}` | byte-level BPE、特殊 token 匹配、解码 | 导入 ASR 词表；核验 Added Tokens 属性与多语言分词 |
| `src/graph.{h,cpp}` | Qwen3 RMSNorm、QK-Norm、GQA、SwiGLU、LM Head | 外部 Embedding 入口、音频图、位置与 mask 校验 |
| `src/kv_cache.{h,cpp}` | 文本 Decoder 的 F16 KV 分配与追加 | 容量包含音频；后续流式另设缓存有效性规则 |
| `src/sampler.{h,cpp}` | 贪心和 EOS 循环基础 | ASR 默认贪心、使用正确 EOS 集合 |
| `src/inference_engine.{h,cpp}` | Init/取消/串行互斥/回调/统计的设计 | 新增音频前处理编排及 ASR 请求响应，不直接复用 Message.content |
| `src/qwen3_chat.{h,cpp}` | 可借鉴模板/解析模块化方法 | ASR 专用模板、语言前缀和输出解析，不复用 Thinking/Tool Call 语义 |
| `tools/convert_hf_to_gguf.py` | GGUF writer、文本张量映射的部分设施 | 注册 ASR、保存音频配置和张量，避免静默漏权重 |

建议后续新增职责边界，而非直接把音频处理塞入 `Qwen3PromptBuilder`：

```text
AudioFrontend       PCM → Mel / 有效长度
AudioEncoder        Mel → [A,1024]
AsrPromptBuilder    context/language/音频长度 → IDs/位置映射
Qwen3Decoder        IDs 或外部 Embedding → KV/logits
AsrOutputParser     token 文本 → language/text
AsrEngine           生命周期、编排、取消、结果和统计
```

以上名字仅为建议，不是已经存在的类。对外可沿用 v1.0 的初始化和取消风格，但 ASR 请求应显式携带 PCM/采样率、context、language 等字段，响应应显式携带 text、language、finish reason 和统计。

GGUF 首版验收必须检查所有 612 个原始存储张量的去向：保留、明确验证后去重或有依据地忽略；不能仅转换 28 层文本 Decoder 就宣称完成整个 ASR 模型。

## 十三、建议的开发与验证顺序

### 13.1 先建立官方基线

固定模型 revision、依赖、dtype 和 attention 后端；先用单条短音频，保存原始 PCM、prompt、IDs、特征长度和生成 token。

特别把“自动语言”和“强制 Chinese”分开保存。当前固定源码的长音频 mask 后端差异未验证前，不用跨窗口 eager 输出充当所有后端通用的金标准。

### 13.2 先验证文本 Decoder 的复用

由官方 Python 导出融合后的 `inputs_embeds`、位置和最后 logits，C++ 先只跑文本 Decoder：

- 逐层对比 hidden state。
- 对比 prefill 最后 logits 和 top-k token。
- 固定相同 token 历史，使用 teacher forcing 对比后续若干步，避免一次 token 分歧使误差难定位。
- 验证原 LLM 路径仍能通过现有示例。

这样能将“音频前端误差”和“Decoder 复用错误”分开。

### 13.3 再补音频图与音频前端

依次比较：

```text
Mel → CNN1 → CNN2 → CNN3 → conv_out
 → 音频各层 → proj2 → 融合 Embedding
 → Decoder prefill → 增量 decode → 最终文本
```

浮点比较同时记录最大绝对误差、RMSE、相对误差和必要时余弦相似度。阈值根据参考精度、F16/BF16 转换和 backend 实测建立，不能随意要求 BF16、F16、F32 全部 bitwise 一致。

### 13.4 最小测试集

| 类别 | 建议覆盖 |
| --- | --- |
| 前处理 | 静音、单频波、短语音；16 kHz 与重采样；单双声道 |
| CNN 边界 | 99/100/101、199/200/201 个有效 Mel 帧，尾块有效长度 |
| 注意力边界 | 小于 8 秒、跨 8 秒、多个窗口，固定参考后端 |
| 模板 | 空/非空 context，自动/指定语言，音频占位数量一致 |
| 语言内容 | 中文、英文、数字、标点、领域词，后续再扩展其他语言 |
| 生成控制 | EOS、max_tokens、n_ctx 溢出、取消、错误返回 |
| 稳定性 | 空输入、异常采样率、NaN/Inf、重复调用不串 KV |
| 扩展能力 | 多音频 batch padding、流式尾块和回改、长音频拼接 |

batch 时两个不同长度音频的输出应与分别推理比较。官方 Transformers 路径内部逐条调用音频塔以保持精度，不应因 batch 外层存在就假设音频塔可任意混合打包。

### 13.5 性能统计

建议拆开记录：音频解码/重采样、Mel、音频 encoder、Decoder prefill、decode、解析的耗时。

```text
端到端时延 = 前处理 + 音频编码 + 文本 prefill + decode + 后处理
RTF = 处理耗时 / 原始音频时长
```

`RTF<1` 表示处理速度快于实时，但不等于流式首字延迟低。自动语言模式的第一个生成 token 可能只是 `language` 元信息，宜分别记录首模型 token 和首可见转写文字时延。离线测量通常不含用户录音等待时间，应明确统计边界。

推荐实现顺序总结：**官方短音频基线 → 外部 Embedding 驱动现有 Decoder → 音频 encoder → C++ Mel → 离线端到端 → 长音频与 batch → 真正音频流式**。量化和其他性能优化放在数值对齐之后。
