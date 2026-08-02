# Qwen3-Embedding-0.6B 模型架构与推理详解

> 阅读前提：你已经跟着 `Qwen3-0.6B_Model_Architecture_And_Inference.md` 走过一遍 LLM 推理，
> 熟悉 Embedding → RoPE → Attention → KV Cache → MLP → LM Head → 采样 这条链路。
> 本文聚焦：**Embedding 推理与 LLM 推理到底哪里一样、哪里不一样**，帮你把已有经验迁移过来。

---

## 一、最终目标

学完本文，你应该能够：

- 说清楚「文本 → 句向量」的完整数据流
- 理解为什么 Embedding 推理**只需要一次前向、不需要自回归、不需要采样**
- 理解 **last-token pooling** 和 **L2 归一化** 的作用
- 理解 query 的 **Instruct 指令前缀**，以及它和 LLM chat 模板的区别
- 复用你为 Qwen3-0.6B 写的 Transformer 前向图，改造出一个最小 Embedding Demo
- 理解 embedding 之间怎么算相似度（余弦 = 点积）

---

## 二、LLM 推理 vs Embedding 推理（全景对比）

``` text
        LLM 推理（生成）                       Embedding 推理（表示）
   ───────────────────────────          ─────────────────────────────
   1. 编码 prompt                        1. 编码文本（query 加 Instruct 前缀）
   2. Prefill（一次前向，建 KV Cache）    2. 一次前向（无需保留 KV 供后续复用）
   3. 取最后位置 logits → 采样一个 token  3. 取最后位置 hidden_state（不算 logits）
   4. Decode：把新 token 喂回，循环 2~3   4. ✗ 没有循环
   5. 命中 EOS 停止，得到一串 token       5. Pooling + Normalize，得到一个向量
```

一句话：

> **LLM 是「多次前向、逐字吐 token」；Embedding 是「一次前向、吐一个向量」。**

因此 Embedding 推理里**天然消失**的东西：

- ❌ 自回归 decode 循环
- ❌ 采样器（temperature / top-k / top-p / repetition penalty）
- ❌ KV Cache 的“逐步增长 / 复用”（单次前向内部仍有注意力，但无需跨步缓存）
- ❌ EOS 停止判断（长度就是输入长度，跑完即止）
- ❌ LM Head / Softmax over vocab

**保留并完全复用**的东西：

- ✅ Tokenizer（Qwen2 BPE）
- ✅ Embedding 查表
- ✅ RoPE
- ✅ 28 层 Transformer（GQA Attention + SwiGLU + RMSNorm）
- ✅ 因果注意力 mask（Embedding 模型依旧是 causal 的）

---

## 三、逐章拆解

### 第 1 章：整体数据流

``` text
文本
 │  ① Tokenize（+ 末尾 <|endoftext|>）
token ids
 │  ② Embedding 查表 → (SeqLen, 1024)
 │  ③ + RoPE
 │  ④ Transformer ×28（causal attention）
 │
last_hidden_state (SeqLen, 1024)
 │  ⑤ Pooling：取最后一个 token → (1024)
 │  ⑥ L2 Normalize → (1024, 模长=1)
句向量 embedding
```

⑤⑥ 是与 LLM 唯一不同的两步。前面 ①~④ 与你写过的 LLM 前向**逐算子一致**。

---

### 第 2 章：输入层（Tokenizer + Instruct 提示词）

Embedding 模型的输入分两类，处理方式不同：

| 类型 | 是否加指令 | 格式 |
| ---- | ---- | ---- |
| **query（查询）** | ✅ 加 | `Instruct: {task}\nQuery:{query}` |
| **document（被检索文档）** | ❌ 不加 | 原文直接输入 |

`config_sentence_transformers.json` 里预置了 query 模板：

``` json
{
  "prompts": {
    "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    "document": ""
  },
  "similarity_fn_name": "cosine"
}
```

举例，一条查询实际喂给模型的文本是：

``` text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:What is the capital of China?
```

而文档端只输入：

``` text
The capital of China is Beijing.
```

> **与 LLM chat 模板的区别**：
> - LLM chat 模板是 `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`，目的是“让模型开始回答”。
> - Embedding 的 Instruct 前缀不是对话，而是“告诉模型这次检索任务是什么”，让向量更贴合任务。
> - 官方建议：指令用**英文**写、按任务/语言定制，通常能带来 1%~5% 的检索提升；文档端**不要**加指令。

**编码收尾**：Qwen3-Embedding 会在 token 序列末尾追加 `<|endoftext|>`(151643) 作为“句尾锚点”，
last-token pooling 取的就是这个位置的 hidden state。

---

### 第 3 章：RoPE 位置编码

与 LLM 完全相同：`rope_theta = 1000000`，同样的旋转方式。
唯一相关的差异是 `max_position_embeddings = 32768`（LLM 是 40960），即官方对外的 32K 上下文。
实现上无需任何改动，直接复用。

---

### 第 4 章：Attention（依旧是 GQA + 因果 mask）

参数与 LLM 一字不差：

| 参数 | 值 |
| ---- | ---- |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| attention_bias | false |

**重点：Embedding 模型仍然使用因果掩码（causal mask）。**
这就是为什么 pooling 要取**最后一个** token——只有最后一个位置能注意到（attend to）整句话，
它的输出向量才包含全句语义。

> 单次前向内部，Attention 当然还是要算 Q·Kᵀ、Softmax、乘 V；
> 但因为不做逐 token 生成，**不需要把 K/V 缓存起来供“下一步”复用**，
> 所以工程上可以不实现“KV Cache 增量写入 / 跨步读取”那套逻辑，一次性算完 SeqLen 个位置即可。

---

### 第 5 章：为什么 Embedding 不需要 KV Cache（对比 LLM）

回顾 LLM：KV Cache 的价值在于 **decode 阶段**——每生成一个新 token，
只算这一个新位置的 Q，然后去缓存里读历史 K/V，避免重复计算前面所有 token。

Embedding 没有 decode 阶段：

``` text
LLM：  Prefill(N) → decode(1) → decode(1) → ... 反复 → 需要 KV Cache 省算力
Embed：Forward(N) 一次算完 → 结束            → 无“后续步”，KV 无处复用
```

所以 Embedding 推理的内存模型比 LLM **简单得多**：
没有随生成变长的 cache，显存/内存占用在一次前向后就固定了。

---

### 第 6 章：MLP（SwiGLU）与 第 7 章：RMSNorm

与 LLM 完全一致，无需改动：

- MLP：`1024 → 3072(SwiGLU) → 1024`，激活 `silu`。
- RMSNorm：`rms_norm_eps = 1e-6`。

直接复用你已有的算子实现。

---

### 第 8 章：输出层（本文核心差异）

这是**唯一需要新写**的部分，替代 LLM 的「LM Head → Softmax → 采样」。

#### 8.1 Pooling：last-token

从 `last_hidden_state`（形状 `SeqLen × 1024`）里取**最后一个有效 token**那一行：

``` python
# 官方参考实现（Transformers）
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]          # 左填充：末尾就是最后 token
    else:
        seq_len = attention_mask.sum(dim=1) - 1    # 右填充：按真实长度取
        bs = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(bs), seq_len]
```

- 若用 `padding_side='left'`（官方推荐），直接 `[:, -1]`。
- 单条推理、无 padding 时，就是 `hidden[-1]`，最简单。

#### 8.2 L2 Normalize

``` python
embeddings = F.normalize(embeddings, p=2, dim=1)   # 每个向量除以自身模长
```

归一化后 `‖v‖ = 1`，于是：

``` text
cos(a, b) = (a · b) / (‖a‖‖b‖) = a · b   （因为模长都=1）
```

检索时直接做**点积**即可得到余弦相似度。

#### 8.3 相似度计算

``` python
scores = query_embeddings @ document_embeddings.T
# 官方示例输出：
# [[0.7646, 0.1414],
#  [0.1355, 0.6000]]
```

对角线（同主题的 query-doc 对）分数明显更高，说明向量语义对齐正确。

---

## 四、最小推理 Demo 伪代码（复用 LLM 骨干）

``` cpp
// 复用你为 Qwen3-0.6B 写好的部分：tokenizer、embedding、rope、28 层 transformer 图
std::vector<int> ids = tokenizer.encode(add_query_instruct(text));  // query 加 Instruct 前缀
ids.push_back(EOS_ENDOFTEXT);                 // 末尾 <|endoftext|>(151643)

// 一次前向（无 decode 循环、无采样、无 KV 复用）
Tensor h = transformer_forward(ids);          // (SeqLen, 1024)，与 LLM 的 prefill 同一段图

// —— 以下是 Embedding 专属的两步 ——
Tensor v = h.row(SeqLen - 1);                 // last-token pooling（左填充/单条时取末行）
v = l2_normalize(v);                          // (1024)，模长=1

return v;                                      // 句向量
```

对比 LLM Demo：把「取 logits → 采样 → 喂回循环」整段换成「取末行 → 归一化」即可。

---

## 五、给你的小测试

能回答以下 5 个问题，说明你已吃透 Embedding 推理：

1. 为什么 Embedding 推理不需要采样器和 EOS 停止判断？
2. 为什么 pooling 取的是**最后一个** token 而不是第一个或平均？（提示：因果 mask）
3. 为什么 Embedding 模型工程上可以不实现 KV Cache 的增量读写？
4. query 要加 Instruct 前缀、document 不加，分别是为什么？
5. 为什么做了 L2 归一化之后，余弦相似度可以直接用点积算？

> 参数配置细节见 `Qwen3-Embedding-0.6B_Model_Architecture_And_Configs.md`；
> 模型目录文件说明见 `Qwen3-Embedding-0.6B_Resource.md`。
