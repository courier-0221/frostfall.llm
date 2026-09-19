# Qwen3-ASR-0.6B 模型架构与配置详解

## 一、文档基线与核心结论

本文延续 [Qwen3-0.6B 架构与配置](Qwen3-0.6B_Model_Architecture_And_Configs.md) 的模块化说明方式，但参数以 ASR 原始模型包及官方源码为准，不沿用其他 Qwen3 模型的参数默认值。

模型与源码固定版本、下载入口见 [资源文档](Qwen3-ASR-0.6B_Resource.md)。本文主要依据：

- [原始模型 config.json](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/config.json)。
- [官方 modeling_qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)。
- [官方 Processor](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/processing_qwen3_asr.py)。
- 官方 `model.safetensors` 文件头的实际张量形状。

先记住三点：

1. 整体是“音频编码器 + 连续特征融合 + 自回归文本 Decoder”，不是纯文本 Decoder Only 模型。
2. 文本 Decoder 的主要尺寸与 Qwen3-0.6B 相同；复用计算逻辑不等于可以复用现有 LLM 权重。
3. 音频侧使用 CNN、LayerNorm、GELU、非因果注意力；文本侧使用 RMSNorm、QK-Norm、RoPE、GQA、SwiGLU 和因果注意力。

## 二、整体结构

```text
音频文件 / PCM
      │
单声道、16 kHz、float32
      │
STFT → 128 维 Log-Mel                       system context / ASR 模板
      │                                              │
按 100 个 Mel 帧分块                           Qwen2 风格 BPE
      │                                              │
3 × Conv2d(kernel=3, stride=2, padding=1)          token IDs
每次卷积后 GELU                                      │
      │                                         Embedding 查表
展平 channel × frequency                            │
conv_out: 7680 → 896                                 │
      │                                              │
加分块内正弦位置编码                                 │
      │                                              │
音频 Transformer × 18                               │
      │                                              │
LayerNorm → Linear(896→896) → GELU                   │
      │                                              │
Linear(896→1024)                                     │
      │                                              │
连续音频特征 [A,1024] ─── 替换 audio_pad 位置 ─────────┘
                             │
                    混合输入 [S,1024]
                             │
                   Qwen3 Decoder × 28
                   每层对 Q/K 应用 RoPE
                             │
                       Final RMSNorm
                             │
                  LM Head: 1024 → 151936
                             │
                    贪心自回归生成 token
                             │
                 language / <asr_text> / 转写正文
```

这里没有 Cross-Attention 层。音频特征进入 Decoder 输入序列后，与其他位置一起参与因果 Self-Attention，并产生各层 KV。

`thinker` 是官方组合模块的名字，不表示推理时必须输出 `<think>`；ASR 官方模板不附加聊天模型的 Thinking 前缀。

## 三、配置树：参数分别属于谁

```text
config.json
├── architectures: [Qwen3ASRForConditionalGeneration]
├── model_type: qwen3_asr
├── support_languages: 30 种语言名称
└── thinker_config
    ├── audio_config       音频编码器
    ├── text_config        文本 Decoder
    ├── audio_start_token_id: 151669
    ├── audio_end_token_id:   151670
    ├── audio_token_id:       151676
    └── dtype: bfloat16
```

`config.json` 中还包含许多通用 `PretrainedConfig` 字段，如音频子配置里的 `max_length=20`、`num_beams=1`。这些字段不能被当成音频编码器的时间长度或本任务生成上限。

类似地，文本子配置保存的 `is_decoder=false` 并不能否定实际因果 Decoder 结构；实际 `forward` 使用 causal mask 和自回归 `generate`。判断结构要同时看权重包与执行代码。

## 四、音频前处理配置

来自 [preprocessor_config.json](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/preprocessor_config.json) 及其引用的 `WhisperFeatureExtractor`：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `feature_extractor_type` | `WhisperFeatureExtractor` | 使用 Whisper 风格特征提取，不等于使用 Whisper 网络 |
| `sampling_rate` | 16000 | 原始 JSON 未显式列出；Processor 和特征提取器默认值一致 |
| `feature_size` / `num_mel_bins` | 128 | 每个时间帧的 Mel 频带数 |
| `n_fft` | 400 | 25 ms FFT 窗口 |
| `hop_length` | 160 | 10 ms 帧移，约 100 帧/秒 |
| `dither` | 0.0 | 不加随机抖动噪声 |
| `chunk_length` | 30 | 特征提取器默认长度参数，不是 ASR 硬上限 |
| `n_samples` / `nb_max_frames` | 480000 / 3000 | 对应 30 秒的默认配置值 |
| `padding_side` | `right` | 波形/音频特征向右 padding |
| `return_attention_mask` | true | 用有效帧数排除 padding |

官方 Processor 调用时明确设置 `padding=True`、`truncation=False`，因此不能实现为“固定截取前 30 秒”。文本 batch 使用左 padding；音频和文本 padding 方向不同。

前端输出通常表示为 `[B,128,T_pad]`，有效长度由 `feature_attention_mask` 确定。单样本进入音频塔时先裁剪为 `[128,T]`，T 不包含 batch padding。

## 五、音频编码器参数与形状

### 5.1 关键配置

| `thinker_config.audio_config` 参数 | 实际值 | 作用 |
| --- | ---: | --- |
| `d_model` | 896 | 音频 Transformer 隐藏宽度 |
| `encoder_layers` | 18 | 编码层数 |
| `encoder_attention_heads` | 14 | 标准多头注意力，Q/K/V 均为 14 头 |
| 每头维度，推导值 | 64 | `896 / 14` |
| `encoder_ffn_dim` | 3584 | 普通两层 FFN 中间宽度 |
| `activation_function` | `gelu` | 非 SwiGLU |
| `downsample_hidden_size` | 480 | 三层 Conv2d 的输出通道数 |
| `output_dim` | 1024 | 对齐文本 Decoder 的输入宽度 |
| `num_mel_bins` | 128 | 频率输入维度 |
| `n_window` | 50 | CNN 实际分块长度为 `2 × n_window = 100` Mel 帧 |
| `n_window_infer` | 800 | 构建编码器推理注意力分块边界的参数 |
| `conv_chunksize` | 500 | 一次执行卷积的分块数量上限，不是 500 ms |
| `max_source_positions` | 1500 | 正弦位置表长度，不是整段音频最大帧数 |
| `scale_embedding` | false | 当前未启用 `sqrt(d_model)` 缩放 |
| dropout 各项 | 0 | 推理不做 dropout |

### 5.2 三层二维卷积

以下采用 PyTorch 顺序 `[N,C,F,T]`；N 是音频内部 CNN 分块数，不是服务请求 batch size。以完整的 100 帧块为例：

| 步骤 | 输入 → 输出 | 权重形状，PyTorch 顺序 |
| --- | --- | --- |
| Conv2d1 + GELU | `[N,1,128,100] → [N,480,64,50]` | `[480,1,3,3]`，有 bias |
| Conv2d2 + GELU | `[N,480,64,50] → [N,480,32,25]` | `[480,480,3,3]`，有 bias |
| Conv2d3 + GELU | `[N,480,32,25] → [N,480,16,13]` | `[480,480,3,3]`，有 bias |
| permute + flatten | `[N,480,16,13] → [N,13,7680]` | 展平顺序为 channel 再 frequency |
| `conv_out` | `[N,13,7680] → [N,13,896]` | `[896,7680]`，无 bias |

`kernel=3,stride=2,padding=1` 同时作用于频率和时间维。不能用一层一维卷积替代，也不能只压缩时间维。

### 5.3 音频 token 数不是简单的 `ceil(T/8)`

三次 stride=2 看似 8 倍下采样，但官方先把音频按每 100 帧切块，各块独立卷积。因此完整块的 100 帧变成 13 个位置。

令 `T=100q+r`，`0≤r<100`：

```text
A(T) = 13q + ceil(r / 8)
```

等价于官方 `_get_feat_extract_output_lengths` 的整数计算；`r=0` 时尾项为 0。

| 有效 Mel 帧数 T | 音频位置数 A |
| ---: | ---: |
| 50 | 7 |
| 99 | 13 |
| 100 | 13 |
| 101 | 14 |
| 200 | 26 |
| 3000，约 30 秒 | 390 |

所以 30 秒不是 375 个位置。架构可以描述为约 12.5 Hz 的 8 倍下采样，但按本实现的 100 帧分块与取整，完整秒块实际是 13 个位置/秒。音频占位符必须采用 Processor 的准确长度公式。

对于尾块，源码先在 Mel 空间补齐到本次最长块，卷积结束后根据有效输出长度去掉 padding 位置。不能先把各个尾块都单独按原长卷积，再假设边界数值必然相同。

### 5.4 音频位置编码

音频侧使用固定正弦位置编码，宽度为 896：

```text
freq[i] = exp(-log(10000) × i / (448 - 1))
PE[p]   = concat(sin(p × freq), cos(p × freq))
```

位置向量加在 `conv_out` 输出上。每个 CNN 分块从局部位置 0 重新开始；完整块为 0…12，而不是对整段音频连续编号。该表由公式生成、作为非持久 buffer 保存，不是需要从 safetensors 加载的训练权重。

### 5.5 音频 Transformer Block × 18

```text
X [A,896]
  ├── LayerNorm → 多头非因果 Self-Attention → 残差相加
  └── LayerNorm → Linear(896→3584) → GELU
                → Linear(3584→896) → 残差相加
```

注意力内部：

- Q/K/V 投影均为 `896→896`，均带 bias；输出投影也带 bias。
- Q/K/V 逻辑形状为 `[14,A,64]`。
- 缩放系数是 `1/sqrt(64)=1/8`。
- 音频侧没有文本侧的 QK RMSNorm 和 RoPE。
- LayerNorm 有均值中心化、weight 和 bias；源码使用 `nn.LayerNorm` 默认 `eps=1e-5`。
- GELU 对齐官方实现，不能换成 SiLU 或未验证的近似形式。

注意力分块与参考后端必须一起核对：

1. 音频塔根据 `cu_seqlens` 描述非因果分块。完整块条件下，`window_aftercnn = 13 × (800/100) = 104`，约对应 8 秒音频。
2. FlashAttention 2 的 varlen 路径使用这些边界：块内双向可见、块间隔离。这不是滑动窗口，也不是 Decoder 因果 mask。
3. **本文固定 commit 的 Transformers 实现存在后端路径差异**：定义了 `_prepare_attention_mask`，但音频塔调用 encoder layer 时仅传入 `cu_seqlens`，没有调用该 helper；其自定义 eager attention 不读取 `cu_seqlens`。从代码看，多块音频的 eager 路径可能退化为全局双向注意力。
4. 因而不能声称此版本各后端对长音频严格等价。移植时先用单窗口短音频做数值对齐，再明确选择并验证长音频参考后端；若复现分块语义，ggml 应显式构建块对角双向 mask，或逐注意力块计算。

第 3 点是静态源码观察，本次未运行多后端差分实验。更换官方版本后应重新核验，不能把该现象推广到所有版本。

### 5.6 输出投影

18 层后执行：

```text
[A,896]
 → ln_post: LayerNorm
 → proj1: Linear(896→896)，有 bias
 → GELU
 → proj2: Linear(896→1024)，有 bias
 → [A,1024]
```

这就是送入 LLM 的连续音频特征；不存在额外的音频码本量化步骤，也不能直接用检索 embedding 模型的句向量替代。

## 六、文本 Decoder：与现有 Qwen3-0.6B 对照

### 6.1 配置表

| 参数 | ASR 文本子模型 | 当前 Qwen3-0.6B 文档基线 | 含义 |
| --- | --- | --- | --- |
| `hidden_size` | 1024 | 1024 | 残差流宽度 |
| `num_hidden_layers` | 28 | 28 | Decoder 层数 |
| `num_attention_heads` | 16 | 16 | Q 头数 |
| `num_key_value_heads` | 8 | 8 | KV 头数，2 个 Q 头共享一组 KV |
| `head_dim` | 128 | 128 | 独立超参数，不是 `1024/16` |
| `intermediate_size` | 3072 | 3072 | SwiGLU 中间宽度 |
| `hidden_act` | `silu` | `silu` | 门控分支激活 |
| `rms_norm_eps` | 1e-6 | 1e-6 | RMSNorm 稳定项 |
| `attention_bias` | false | false | 文本投影不带 bias |
| `vocab_size` | 151936 | 151936 | Embedding/LM Head 行数，词表内容仍需核对 |
| `max_position_embeddings` | 65536 | 40960 | 模型位置配置；不等于运行时必须分配的 n_ctx |
| `rope_theta` | 1000000 | 1000000 | RoPE 频率基数 |
| `rope_scaling` | interleaved MRoPE 配置 | null | ASR 默认三个位置轴相同，详见下节 |
| `use_cache` | true | true | Decoder 自回归 KV Cache |
| `tie_word_embeddings` | 子配置为 true | true | ASR 文件实际保存了两份矩阵，不能只据配置省略权重 |

`16×128=2048` 是 Q 注意力内部宽度，与 `hidden_size=1024` 不同。这来自独立的 `head_dim` 设计；GQA 解释的是 Q/KV 头数关系，不是产生这一宽度差异的必要条件。

### 6.2 一个 Decoder Block 的计算

以当前前向包含 S 个位置为例，省略 batch 维：

| 步骤 | 形状变化 |
| --- | --- |
| 输入 RMSNorm | `[S,1024] → [S,1024]` |
| Q 投影 | `[S,1024] → [S,2048] → [S,16,128]` |
| K/V 投影 | 各自 `[S,1024] → [S,1024] → [S,8,128]` |
| QK-Norm | 对每个 head 的 128 维分别做 RMSNorm |
| RoPE | 仅旋转 Q 和 K，不旋转 V，也不直接加到输入 Embedding |
| KV 写入与 GQA | 当前 Q 读取历史和本次合法位置的 K/V |
| 注意力输出投影 | `[S,16,128] → [S,2048] → [S,1024]` |
| 残差 | 与 block 输入相加 |
| FFN RMSNorm | `[S,1024]` |
| gate/up | 两个并行的 `1024→3072` 投影 |
| SwiGLU | `SiLU(gate) × up`，再 `3072→1024` |
| 第二次残差 | 输出仍为 `[S,1024]` |

28 层后再做 Final RMSNorm 和 LM Head，得到 `[S,151936]` logits。自回归生成只需当前最后一个位置的 logits；如在 ggml 中只投影最后一行，可以避免生成整个 prefill 的巨大 logits 张量。

### 6.3 MRoPE 能否复用现有 NEOX RoPE

ASR 配置：

```json
{
  "rope_theta": 1000000,
  "rope_scaling": {
    "interleaved": true,
    "mrope_interleaved": true,
    "mrope_section": [24, 20, 20],
    "rope_type": "default",
    "type": "default"
  }
}
```

`24+20+20=64=head_dim/2`。官方旋转实现允许三轴位置交织，但默认 ASR 的 `get_rope_index` 将同一序列位置复制三次：

```text
position_ids[0] = position_ids[1] = position_ids[2]
无 padding 单样本时，每轴均为 0,1,2,…,S-1
```

当三个轴相同，交织前后的频率值相同，因此可数学退化为普通一维 RoPE。官方 `rotate_half` 采用前半/后半配对，对应当前 ggml 的 NEOX 旋转语义。

复用条件：使用本原始 ASR 路径、三个轴相同、正确计入音频位置、相同 theta 和 head_dim。batch 左 padding、cache position 和 rope delta 必须正确处理；自定义不同轴位置或换用其他多模态模型时，不能据此省略 MRoPE。

### 6.4 Embedding 与 LM Head

官方文件头实际包含：

```text
thinker.model.embed_tokens.weight [151936,1024]
thinker.lm_head.weight            [151936,1024]
```

它们的共享/去重问题见 [资源文档第四节](Qwen3-ASR-0.6B_Resource.md)。首版应按两个存储张量完整导入；不能因 `text_config.tie_word_embeddings=true` 直接触发现有 loader 的缺失输出权重回退逻辑。

## 七、音频 token 与转写协议

来自 [tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/tokenizer_config.json) 和 [generation_config.json](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/generation_config.json)：

| 文本/配置 | ID | 作用 |
| --- | ---: | --- |
| `<\|endoftext\|>` | 151643 | padding，同时属于生成 EOS 集合 |
| `<\|im_start\|>` | 151644 | ChatML 消息开始 |
| `<\|im_end\|>` | 151645 | ChatML 消息结束，同时属于生成 EOS 集合 |
| `<\|audio_start\|>` | 151669 | 音频区域开始 |
| `<\|audio_end\|>` | 151670 | 音频区域结束，不是文本生成 EOS |
| `<\|audio_pad\|>` | 151676 | 将被连续音频特征替换的占位位置 |
| `<asr_text>` | 151704 | 语言元信息与转写正文分隔；Added Token 的 `special=false` |

表格中的竖线使用 Markdown 转义，实际 token 字面量无反斜杠。自动语言模式的输出形如：

```text
language Chinese<asr_text>今天天气怎么样？
```

`language Chinese` 不是一个固定 special token。`<asr_text>` 虽然是协议标记，但 Tokenizer 将其定义为非 special Added Token，官方 `skip_special_tokens=True` 后仍能看到它。

`audio_pad` 不是应该屏蔽掉的普通 padding；它被替换为音频特征，attention mask 应标记为有效位置。只有真实 batch padding 才应屏蔽。

## 八、面向 ggml 的布局与权重映射检查点

以下是建议映射，不宣称已存在官方 ASR GGUF 规范：

| HF 权重前缀 | 本地复用方向 |
| --- | --- |
| `thinker.model.embed_tokens.weight` | 文本 `token_embd.weight` |
| `thinker.model.layers.i.self_attn.{q,k,v,o}_proj.weight` | 现有 `blk.i.attn_*` |
| `thinker.model.layers.i.self_attn.{q,k}_norm.weight` | 现有 `attn_q_norm` / `attn_k_norm` |
| `thinker.model.layers.i.mlp.*` | 现有 FFN gate/up/down |
| `thinker.model.norm.weight` | 文本 `output_norm.weight` |
| `thinker.lm_head.weight` | 文本 `output.weight`，保留实际矩阵 |
| `thinker.audio_tower.*` | 新增独立音频命名空间，覆盖全部卷积、bias、LayerNorm 与投影 |

PyTorch 线性权重标为 `[out,in]`，ggml 习惯按 `ne=[in,out]` 描述。同样一段连续数据可能不需要实际转置，但必须区分轴命名反转和物理内存重排。CNN 的 channel/frequency/time 展平、QKV reshape 和 GGUF 维度应逐张量核对，不能统一套用“所有权重转置”。

至此可把模型理解为：新增音频前端和音频计算图，扩展 Decoder 的 Embedding 输入入口，再复用现有 Qwen3 层计算与增量生成机制。具体执行顺序见 [推理文档](Qwen3-ASR-0.6B_Model_Architecture_And_Inference.md)。
