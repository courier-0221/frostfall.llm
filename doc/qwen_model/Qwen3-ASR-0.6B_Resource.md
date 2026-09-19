# Qwen3-ASR-0.6B 模型资源与开发基线

## 一、范围与阅读顺序

本文面向在 frostfall.llm v1.0 的 ggml/C++ 推理链路上增加语音识别能力，先梳理资源、真实配置和适配边界，不代表工程已支持 ASR。

建议依次阅读：

1. 本文：资源下载、权重规模、运行环境与开发入口。
2. [模型架构与配置](Qwen3-ASR-0.6B_Model_Architecture_And_Configs.md)：音频编码器、Qwen3 文本解码器、张量形状与配置含义。
3. [端到端推理与 v1.0 适配](Qwen3-ASR-0.6B_Model_Architecture_And_Inference.md)：波形到文字的计算路径、流式语义和验证步骤。

核验基线：

| 项目 | 本文采用的版本/依据 |
| --- | --- |
| 目标模型 | `Qwen/Qwen3-ASR-0.6B`，不带 `-hf` 后缀的原始发布版 |
| ModelScope | 用户指定模型页；其 `config.json` 已与官方 Hugging Face 同名模型交叉核对 |
| HF 模型 revision | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| 官方实现 | `QwenLM/Qwen3-ASR`，commit `7c6daf77a2421100f5fb066495372c00129d39ff` |
| Python 包源码版本 | `qwen-asr==0.0.6`；依赖声明 `transformers==4.57.6` |
| 技术报告 | `arXiv:2601.21337v2` |
| 本地开发基线 | 当前检出的 `llm/v1.0`，源码实际位于 `src/` 根目录 |

用户已完成 LLM、embedding 推理；本次检出的工作区未包含 embedding 引擎源码，因此下面只对当前可见的 LLM v1.0 做具体文件映射，不假设存在 `src/core/`、`src/embed/` 或 `src/llm/`。

## 二、模型定位与能力边界

Qwen3-ASR-0.6B 是音频条件下的自回归文本生成模型：

```text
音频 → Log-Mel → 音频编码器 → 连续音频向量
                                   ↓
文本上下文 → 文本 Embedding → 混合输入序列 → Qwen3 Decoder → 转写文字
```

它不是把 WAV 文件名输入现有聊天模型，也不是 CTC 模型；不需要 CTC blank 合并或 CTC beam search。其音频编码器与文本解码器通过输入向量融合连接，而非经典 Encoder–Decoder Cross-Attention。

| 能力 | 官方范围及工程含义 |
| --- | --- |
| 语音识别、语言识别 | 30 种语言、22 种中文方言，合计官方所称 52 种语言/方言覆盖 |
| 语言控制 | 可自动识别，也可通过 `language="Chinese"` 等指定支持列表中的语言名称 |
| 音频类型 | 语音、歌唱、带 BGM 歌曲；实际效果仍需业务数据验证 |
| 上下文辅助 | `context` 写入 system 段，可提供领域词语背景；不是确定性的热词强制解码 |
| 离线和流式 | 同一模型可用，但原始 `qwen-asr` 包的音频流式接口仅开放在 vLLM 后端 |
| 长音频 | 报告给出单段最长 1200 秒；官方工具可切分更长音频 |
| 时间戳 | 需额外的 `Qwen3-ForcedAligner-0.6B`；ASR token 输出本身不是字词时间戳 |
| 许可证 | Apache-2.0；分发时仍需保留相关许可证与声明 |

不要把 ASR 当作 Qwen3-Omni 全功能模型：本任务是音频转文字，不意味着支持视频、语音合成、说话人分离或现有 LLM 的 Thinking/Tool Calling 协议。

## 三、官方资源入口

| 资源 | 地址 | 用途 |
| --- | --- | --- |
| ModelScope 模型页 | https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B | 国内下载主入口 |
| ModelScope 配置 | https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B/resolve/master/config.json | 直接核对结构参数 |
| HF 原始模型 | https://huggingface.co/Qwen/Qwen3-ASR-0.6B | 模型卡、权重、Tokenizer、Processor 配置 |
| HF 固定 revision | https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 | 本文权重与配置基线 |
| 官方实现 | https://github.com/QwenLM/Qwen3-ASR | Transformers/vLLM 推理、流式、对齐工具 |
| 技术报告 | https://arxiv.org/html/2601.21337v2 | 架构、训练、能力与性能评测 |
| 可选时间对齐模型 | https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B | 输入音频和已有转写，预测时间戳 |
| 原生 Transformers 版本 | https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf | 后续原生集成版本，不与本文原始版混用配置/API |

原始模型包配合 `qwen_asr` 的注册与实现使用。官方后来新增 `-hf` 版本，不能因此认为当前原始权重可直接套用任意版本的 `AutoModelForSpeechSeq2Seq`，也不能混用两套权重命名。

### 3.1 模型目录内需要保留什么

以下文件来自上述 HF revision 的实际文件列表；ModelScope 下载后也应逐项检查。

| 文件 | 作用 | C++/GGUF 适配时的处理 |
| --- | --- | --- |
| `model.safetensors` | 音频塔、文本模型、LM Head 的 BF16 权重 | 建立完整张量清单，转换到 ggml 支持的布局和类型 |
| `config.json` | 嵌套的音频、文本配置及音频 token ID | 不能只读取顶层或把整个模型当作 `qwen3` |
| `preprocessor_config.json` | 128 维 Mel、FFT、hop、padding 等 | 音频前端必须使用同一参数与归一化算法 |
| `generation_config.json` | EOS 集合、padding ID、贪心设置 | 停止条件独立于音频结束标记 |
| `tokenizer_config.json` | Added Tokens、音频标记、Tokenizer 行为 | 必须使用 ASR 自己的配置 |
| `vocab.json`、`merges.txt` | Qwen2 风格 byte-level BPE | 可复用现有 BPE 算法，词表和 merges 重新导入 |
| `chat_template.json` | ASR 专用 ChatML 模板 | 独立于现有聊天模板实现 |
| `README.md` | 模型说明及许可证信息 | 作为资源说明保留 |

这个 revision 未列出 `tokenizer.json` 或权重分片索引；不要把其他模型常见的文件名当作本模型的必需文件。`vocab_size=151936` 是 Embedding 行数，也不能直接等同于已定义 Added Tokens 的数量。

### 3.2 下载示例

以下仅为后续操作示例，本次文档工作未执行整模型下载或安装依赖。

```bash
# 在独立 Python 环境中执行
pip install modelscope
modelscope download --model Qwen/Qwen3-ASR-0.6B \
    --local_dir ./models/Qwen3-ASR-0.6B
```

或者通过 Hugging Face 固定 revision：

```bash
pip install huggingface_hub
hf download Qwen/Qwen3-ASR-0.6B \
    --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
    --local-dir ./models/Qwen3-ASR-0.6B
```

两个命令二选一。ModelScope 与 HF 的 revision 命名不通用；从 ModelScope 下载时另行记录其 revision 和本地文件校验值，不把上述 HF SHA 直接作为 ModelScope revision。

## 四、权重规模：区分模型命名、序列化数量和运行内存

已通过 HTTP Range 读取官方 `model.safetensors` 的文件头，未下载完整权重：

- JSON header 为 `74544` 字节。
- 实际列出 `612` 个权重张量。
- HF API 的 `safetensors.total=938008576` 与按文件头形状求和一致。

| 文件头前缀 | 序列化元素数 | 包含内容 |
| --- | ---: | --- |
| `thinker.audio_tower.*` | 186,376,192 | CNN、18 层音频 Transformer、输出投影 |
| `thinker.model.*` | 596,049,920 | 文本 Embedding、28 层 Decoder、Final RMSNorm |
| `thinker.lm_head.*` | 155,582,464 | `151936 × 1024` 输出矩阵 |
| 合计 | 938,008,576 | 文件内 BF16 元素总数，不等价于独立参数去重后的数量 |

特别注意权重共享：`text_config.tie_word_embeddings=true`，但文件内同时存有 `thinker.model.embed_tokens.weight` 和 `thinker.lm_head.weight`，且数据区间不同。本次小范围抽样中两者开头字节相同，但这不能证明整个矩阵一致。

因此既不能宣称“两个矩阵一定是不同权重”，也不能仅凭嵌套配置删除 LM Head。首版转换应保留两者；后续在完整比较权重并核对实际加载后的共享语义后，才考虑去重。若允许完全共享，序列化合计减去一份矩阵为 `782,426,112`，这只是去重情形下的计算，不是本次实测的独立参数数目。

### 4.1 存储与运行内存估算

按文件内全部元素保留：

```text
BF16/F16 权重数据：938008576 × 2 = 1,876,017,152 字节
约 1.876 GB，或 1.747 GiB
F32 权重数据：约 3.494 GiB
```

以上不包含 GGUF 元数据、Tokenizer、KV Cache、中间激活、分配器和模型转换时的临时副本。F16 与 BF16 同为 2 字节，但数值范围不同，转换后仍需验证精度。

文本 Decoder 的 F16 KV Cache：

```text
每个序列位置 = 28 层 × 2(K/V) × 8 KV heads × 128 head_dim × 2 bytes
             = 114688 bytes = 112 KiB
```

| 容量 | KV 内存，不含其他部分 |
| --- | ---: |
| 4096 个位置 | 448 MiB |
| 8192 个位置 | 896 MiB |
| 65536 个位置 | 7168 MiB，即 7 GiB |

音频位置也占用 Decoder KV。按默认分块规则，30 秒音频约 390 个音频位置，仅这些位置的有效 KV 就约 `42.66 MiB`；若引擎预分配 `n_ctx=4096`，实际仍按 448 MiB 分配。

推荐首版从单条短音频和 `n_ctx=2048/4096` 开始，显式校验：

```text
音频位置数 + 模板/上下文 token 数 + 最大生成 token 数 ≤ n_ctx
```

不要根据 `max_position_embeddings=65536` 无条件分配最大上下文。量化大小也不能只算“参数数 × 4 bit”：块级 scale、未量化张量以及音频塔算子限制都会影响实际文件大小。

## 五、Python 参考推理环境

先建立官方可重复基线，再做 C++ 移植。推荐独立 Python 3.12 环境；官方包声明 Python ≥ 3.9，本文原始版依赖包括 Transformers 4.57.6、Accelerate 1.12.0、librosa、soundfile 等。vLLM 是可选后端，源码声明版本为 0.14.0。

```bash
pip install qwen-asr==0.0.6
```

最小离线转写示例：

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "./models/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=1,
    max_new_tokens=256,
)
results = model.transcribe(audio="./test.wav", language="Chinese")
print(results[0].language)
print(results[0].text)
```

上述示例需要适合 BF16 的 CUDA 环境。CPU 参考可尝试 `device_map="cpu"`、`dtype=torch.float32`，但本次未运行验证，也不承诺实时性能。FlashAttention 2 不是 ggml CPU 运行的依赖；其参考后端的分块行为需与实际选用的注意力后端一起记录，详见架构文档的注意事项。

长音频需要相应增加 `max_new_tokens`；示例的 256 不是模型支持的最大输出长度。

官方报告的 92 ms TTFT 和 128 并发下 2000 秒音频/秒吞吐来自特定 vLLM/CUDA Graph/BF16 测试条件，不是当前 ggml CPU 引擎的性能预期；这两个最优指标也不是同一并发条件下取得。

## 六、GGUF 与当前工程的接入事实

本地核对结果：

- `tools/convert_hf_to_gguf.py` 注册了 `Qwen3ForCausalLM`，没有注册 `Qwen3ASRForConditionalGeneration`。
- `tools/convert_hf_to_gguf_update.py` 也未检出对应 ASR 标识。
- `src/model.cpp` 明确只接受 `general.architecture == "qwen3"`。
- `src/graph.cpp` 入口是 token ID 查表，没有连续音频向量入口。
- `src/qwen3_chat.cpp` 服务于聊天、Thinking 和 Tool Call，不是 ASR 模板。

因此，当前 LLM 的转换命令不能直接视作 ASR 转换方法；修改 architecture 字符串或删掉 `thinker.` 前缀也不能补齐音频计算图。

建议首版采用单模型文件携带完整音频塔和文本权重，并定义清晰的 ASR 元数据、张量命名及版本标识。单文件还是拆分音频/文本文件属于后续实现决策；这里不虚构已经存在的 GGUF 标准架构名、转换脚本或命令。

## 七、官方源码阅读索引

以下链接固定到本文核验的 commit：

| 文件 | 阅读重点 |
| --- | --- |
| [configuration_qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/configuration_qwen3_asr.py) | 配置层级；类默认参数不等于 0.6B 权重包实际参数 |
| [modeling_qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py) | CNN、音频编码、特征替换、文本 Decoder 与位置编码 |
| [processing_qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/processing_qwen3_asr.py) | Log-Mel、音频占位符扩展、长度公式 |
| [qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py) | `transcribe`、语言前缀、离线/流式编排 |
| [utils.py](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/utils.py) | 音频标准化、长音频切分、转写解析 |
| [WhisperFeatureExtractor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/feature_extraction_whisper.py) | STFT、Slaney Mel、对数压缩、padding |

## 八、本次已核验与尚未验证

已核验官方配置、实际权重文件头、部分权重字节、官方推理源码和本地 LLM v1.0 入口。未下载完整模型、未执行真实音频推理、未生成 GGUF、未实现 C++ ASR，也未测量本机准确率和吞吐。

后续开始实现前，应保存模型校验值、参考后端/精度/依赖版本，以及短音频的中间张量与输出，作为逐模块对齐基准。
