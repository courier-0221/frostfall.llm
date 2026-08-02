# Qwen3-Embedding-0.6B 模型架构与 config.json 参数详解

> 阅读前提：本文假设你已经读过 `Qwen3-0.6B_Model_Architecture_And_Configs.md`，
> 因此对 Decoder Only Transformer、GQA、RoPE、RMSNorm、SwiGLU 已经熟悉。
> 本文只做一件事：**在你熟悉的 Qwen3-0.6B 之上，讲清楚 Embedding 模型多了什么、少了什么、改了什么。**

---

## 一、一句话认识 Embedding 模型

Qwen3-Embedding-0.6B 与 Qwen3-0.6B（LLM）**共用同一套 Transformer 骨干**
（同样 28 层、hidden_size=1024、16 Q head + 8 KV head、SwiGLU、RoPE），
区别在于**它不再预测下一个 token，而是把整段文本压缩成一个 1024 维向量**。

``` text
        LLM  (Qwen3-0.6B)                 Embedding (Qwen3-Embedding-0.6B)
   ─────────────────────────         ─────────────────────────────────────
        token ids                             token ids
            │                                     │
        Transformer ×28                       Transformer ×28     ← 骨干完全一样
            │                                     │
      last hidden state                    last hidden state
            │                                     │
        LM Head (→ 151936 logits)          ✗ 丢弃 LM Head
            │                                     │
        Softmax → 采样下一个 token          Pooling(取最后一个 token)
            │                                     │
        自回归循环生成                        L2 Normalize
                                                  │
                                          1024 维句向量（embedding）
```

一句话总结：

> **LLM = Transformer + LM Head + 自回归采样**
> **Embedding = Transformer + Pooling + Normalize（单次前向、无采样、无 KV Cache 增长）**

---

## 二、Qwen3-Embedding-0.6B 整体结构

``` text
                 输入文本
        （query 需加 Instruct 指令前缀）
                     │
             Tokenizer 编码
                     │
             token id (int)  + 末尾 <|endoftext|>
                     │
──────────────────────────────────────────────
              输入层（Embedding）
──────────────────────────────────────────────

      vocab_size = 151669           ← 与 LLM 的 151936 不同
                │
        Embedding Matrix
      (151669 × 1024)
                │
      hidden_size = 1024
                │
        加 RoPE 位置编码 (theta=1000000)
                │
──────────────────────────────────────────────
        Transformer Block × 28       ← 与 LLM 完全一致
──────────────────────────────────────────────

 Block0 ┌────────────────────────┐
        │ RMSNorm                │
        │ Multi Head Attention   │  (GQA: 16 Q / 8 KV, head_dim=128)
        │ Residual               │
        │ RMSNorm                │
        │ MLP(SwiGLU)            │
        │ Residual               │
        └────────────────────────┘
 Block1 ...
 Block27

──────────────────────────────────────────────
              输出层（关键差异）
──────────────────────────────────────────────

      last_hidden_state  (SeqLen × 1024)
                │
        Pooling：取最后一个 token  ← last-token pooling
                │
          向量 (1024)
                │
        L2 Normalize（除以自身模长）
                │
      最终 embedding (1024, 模长=1)
```

对比：LLM 到这里是 `RMSNorm → LM Head → logits(151936) → Softmax → 采样`，
Embedding 模型则是 `Pooling → Normalize → 句向量`。

---

## 三、config.json 逐字段对比（重点看差异）

Embedding 模型的 `config.json` 内容如下：

``` json
{
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151669
}
```

### 与 Qwen3-0.6B (LLM) 的字段对比表

| 参数 | Qwen3-0.6B (LLM) | Qwen3-Embedding-0.6B | 是否相同 |
| ---- | ---- | ---- | ---- |
| architectures | Qwen3ForCausalLM | Qwen3ForCausalLM | ✅ 相同（见下方说明） |
| model_type | qwen3 | qwen3 | ✅ |
| hidden_size | 1024 | 1024 | ✅ |
| num_hidden_layers | 28 | 28 | ✅ |
| num_attention_heads | 16 | 16 | ✅ |
| num_key_value_heads | 8 | 8 | ✅ |
| head_dim | 128 | 128 | ✅ |
| intermediate_size | 3072 | 3072 | ✅ |
| hidden_act | silu | silu | ✅ |
| rms_norm_eps | 1e-6 | 1e-6 | ✅ |
| rope_theta | 1000000 | 1000000 | ✅ |
| tie_word_embeddings | true | true | ✅ |
| attention_bias | false | false | ✅ |
| **vocab_size** | **151936** | **151669** | ❌ 不同 |
| **max_position_embeddings** | **40960** | **32768** | ❌ 不同 |
| bos_token_id | 151643 | 151643 | ✅ |
| eos_token_id | 151645(`<im_end>`)* | 151643(`<endoftext>`) | ❌ 不同 |
| torch_dtype | bfloat16 | bfloat16 | ✅ |

> \* LLM 的实际停止 token 是 `<|im_end|>`(151645)；Embedding 的 `config.json` / `generation_config.json`
> 里 eos 是 `<|endoftext|>`(151643)，因为 embedding 的输入末尾要拼接 `<|endoftext|>` 作为“句尾锚点”。

### 关键结论

1. **骨干 100% 一致**：从 `hidden_size` 到 `rope_theta` 这一整套决定“网络长什么样”的参数完全相同。
   这意味着你为 Qwen3-0.6B 写的 Transformer 前向图（graph）几乎可以**原样复用**。
2. **只有三处实质差异**：
   - `vocab_size`：151936 → **151669**（Embedding 用了裁剪后的词表，Embedding Matrix 尺寸随之变小）。
   - `max_position_embeddings`：40960 → **32768**（最大上下文，官方对外宣传 32K）。
   - `eos_token_id`：语义不同（见上）。

---

## 四、`architectures` 仍是 `Qwen3ForCausalLM`，为什么？

这是最容易困惑的一点：明明是 Embedding 模型，`architectures` 却写着 `Qwen3ForCausalLM`（因果语言模型）。

原因：

- Qwen3-Embedding 是在 **Qwen3-0.6B-Base 因果模型**基础上，用对比学习（contrastive learning）微调而来。
- 权重结构、config 都沿用因果模型的定义（甚至保留了 causal mask 掩码机制）。
- **“它是 embedding 模型”这件事不写在 config.json 里，而是写在 sentence-transformers 的组装文件里**
  （`modules.json` / `1_Pooling/config.json` / `config_sentence_transformers.json`，详见 Resource 文档）。

也就是说：

``` text
config.json          → 只描述“Transformer 骨干”（还是那个因果模型）
modules.json 等      → 描述“骨干之后接 Pooling + Normalize”，把它变成 embedding 模型
```

对推理引擎的启示：**你加载权重、搭 Transformer 图的代码可以和 LLM 共用；
Embedding 特有的逻辑（Pooling / Normalize / Instruct 前缀）在骨干之外单独实现即可。**

---

## 五、输出层：从「LM Head」到「Pooling + Normalize」

这是 Embedding 模型唯一需要重写的部分。

### 5.1 丢弃 LM Head

LLM 在最后一层 hidden state 上乘 `Embedding^T`（tie_word_embeddings）得到 151936 维 logits。
**Embedding 模型不需要 logits**，直接拿 `last_hidden_state`（形状 `SeqLen × 1024`）。

> 注意：`tie_word_embeddings=true` 依旧成立，但那份权重只在“输入 Embedding”方向用得到，
> 输出方向（LM Head）在 embedding 推理里根本不跑。

### 5.2 Pooling：last-token（取最后一个 token）

`1_Pooling/config.json` 指定：

``` json
{
  "word_embedding_dimension": 1024,
  "pooling_mode_lasttoken": true,     ← 只用最后一个 token
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": false,
  "include_prompt": true
}
```

即：**在 `SeqLen × 1024` 里，只取“最后一个有效 token”那一行 1024 维向量。**

为什么是最后一个 token？因为骨干是 **因果注意力（causal attention）**——
只有最后一个位置能“看到”整句话的全部信息，所以它的 hidden state 天然是整句的语义汇总。
（对比：BERT 类双向模型常用 CLS 或 mean pooling。）

配合 `padding_side='left'`（左填充），最后一个 token 永远在序列末尾，取 `[:, -1]` 即可。

### 5.3 L2 Normalize

`modules.json` 的第 3 个模块是 `Normalize`：把向量除以自身的 L2 模长，使 `‖v‖ = 1`。

好处：归一化后，两个向量的**余弦相似度 = 点积**，检索时直接做内积即可，又快又简单
（`config_sentence_transformers.json` 里 `similarity_fn_name: "cosine"`）。

---

## 六、完整结构图（Embedding 版）

``` text
                 输入文本
        query: "Instruct: {task}\nQuery:{q}"
        document: 原文（不加指令）
                     │
        Tokenizer（词表 151669）
        末尾追加 <|endoftext|>(151643)
                     │
        Embedding Matrix (151669 × 1024)
                     │
        + RoPE(theta=1000000)
                     │
──────────────────────────────────────────
        Transformer ×28  （与 LLM 完全相同）
   RMSNorm → GQA(16Q/8KV, dim=128) → Residual
        → RMSNorm → SwiGLU(1024→3072→1024) → Residual
──────────────────────────────────────────
                     │
        last_hidden_state (SeqLen × 1024)
                     │
        Pooling: last-token  →  (1024)
                     │
        L2 Normalize         →  (1024, 模长=1)
                     │
              句向量 embedding
                     │
        与其他 embedding 做余弦相似度 = 点积
```

---

## 七、小结：从 LLM 迁移到 Embedding 的改动清单

| 模块 | 是否复用 LLM 代码 | 说明 |
| ---- | ---- | ---- |
| Tokenizer | ✅ 复用（词表规模改 151669） | 仍是 Qwen2 BPE；注意末尾拼 `<|endoftext|>` |
| Embedding Matrix | ✅ 复用 | 只是行数变 151669 |
| RoPE | ✅ 复用 | theta 相同 |
| Transformer ×28（Attn/MLP/RMSNorm） | ✅ 完全复用 | 参数一字不差 |
| LM Head | ❌ 不跑 | Embedding 不需要 logits |
| Pooling(last-token) | 🆕 新增 | 取最后一个 token 的 hidden state |
| L2 Normalize | 🆕 新增 | 输出单位向量 |
| 采样器 / KV Cache 增长 / 自回归循环 | ❌ 删除 | 单次前向即可，无需逐 token 解码 |
| Instruct/Query 提示词模板 | 🆕 新增 | 替代 LLM 的 chat 模板 |

> 下一篇 `Qwen3-Embedding-0.6B_Model_Architecture_And_Inference.md` 讲推理数据流与手写实现要点；
> `Qwen3-Embedding-0.6B_Resource.md` 讲模型目录里每个文件的作用。
