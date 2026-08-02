# Qwen3-Embedding-0.6B 模型文件解析

> 阅读前提：你已看过 `Qwen3-0.6B_Resource.md`，熟悉 config.json / tokenizer / safetensors 三类资源。
> 本文重点：**Embedding 模型目录比 LLM 多出的一组 sentence-transformers 组装文件**——
> 它们才是把“因果模型骨干”变成“embedding 模型”的关键。

## 目录结构

``` text
Qwen3-Embedding-0.6B
├── config.json                        # 模型结构配置（骨干，与 LLM 同构）
├── configuration.json                 # 仓库管理配置（modelscope 用）
├── generation_config.json             # 生成默认参数（embedding 基本不用）
├── tokenizer.json                     # 完整 Tokenizer 定义
├── tokenizer_config.json              # Tokenizer 行为配置
├── vocab.json                         # 词表
├── merges.txt                         # BPE 合并规则
├── model.safetensors                  # 模型权重（≈1.19GB）
├── README.md                          # 模型说明 / 用法 / Benchmark
│
├── modules.json                       # 🆕 sentence-transformers 流水线定义
├── config_sentence_transformers.json  # 🆕 ST 级配置（prompts / 相似度函数）
├── 1_Pooling/
│   └── config.json                    # 🆕 Pooling 方式（last-token）
└── 2_Normalize/                       # 🆕 L2 归一化（无参数，仅占位）
```

从推理引擎角度，可分为四类资源（比 LLM 多了第四类）：

``` text
Qwen3-Embedding-0.6B
├── 模型结构配置        （config.json 等，与 LLM 同构）
├── Tokenizer 资源      （vocab/merges/tokenizer.json，与 LLM 基本一致）
├── 模型权重            （model.safetensors）
└── 🆕 Embedding 组装配置（modules.json / 1_Pooling / 2_Normalize / config_sentence_transformers.json）
```

> 记住这句：**config.json 只描述 Transformer 骨干；“它是 embedding 模型”这件事写在第四类文件里。**

---

# 一、与 LLM 相同（或几乎相同）的文件

这几类你已经熟悉，这里只标注 Embedding 的差异点。

## 1. config.json

描述 28 层 Transformer 骨干。**与 LLM 结构同构**，仅三处不同：

| 字段 | LLM | Embedding |
| ---- | ---- | ---- |
| vocab_size | 151936 | **151669** |
| max_position_embeddings | 40960 | **32768** |
| eos_token_id | 151645 | **151643** |

`architectures` 依旧是 `Qwen3ForCausalLM`（详见 Configs 文档第四节）。

## 2. tokenizer.json / tokenizer_config.json / vocab.json / merges.txt

仍是 Qwen2 BPE（`tokenizer_class: "Qwen2Tokenizer"`），加载方式与 LLM 一致。
需要注意的几个字段：

``` json
{
  "add_bos_token": false,
  "eos_token": "<|im_end|>",
  "pad_token": "<|endoftext|>",
  "bos_token": null,
  "model_max_length": 131072
}
```

- `add_bos_token: false`：不自动加 BOS。
- `pad_token = <|endoftext|>`：批量推理时用它做 padding。
- Embedding 编码时会在**文本末尾追加 `<|endoftext|>`(151643)**，供 last-token pooling 定位句尾。
- 特殊 token 表（`<|im_start|>` 151644、`<|im_end|>` 151645、`<|endoftext|>` 151643 等）与 LLM 相同。

> `tokenizer_config.json` 里也带了一份 `chat_template`，那是从基座模型继承来的“遗产”，
> **embedding 推理并不用它**——embedding 用的是 Instruct/Query 前缀（见 config_sentence_transformers.json）。

## 3. generation_config.json

``` json
{
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "max_new_tokens": 2048,
  "transformers_version": "4.51.3"
}
```

LLM 里这份文件放采样默认值（temperature/top_p/top_k…）；
**Embedding 不做生成，这份文件几乎用不到**，只是从基座继承下来的占位。

## 4. configuration.json

``` json
{"framework": "pytorch", "task": "text-generation", "allow_remote": true}
```

modelscope/魔搭 仓库管理用。注意 `task` 仍写着 `text-generation`（继承自基座），别被误导——
真正的任务类型由 sentence-transformers 组装文件决定。

## 5. model.safetensors

``` text
≈ 1.19 GB（1,191,586,416 字节）
```

保存 Embedding Matrix + 28 层 Transformer 全部权重。
比 LLM 的 safetensors 略小，因为 **vocab_size 更小（151669 < 151936）** 且**不含独立 lm_head**
（tie_word_embeddings，且 embedding 方向根本不跑输出投影）。加载方式与 LLM 相同。

---

# 二、🆕 Embedding 专属：sentence-transformers 组装文件

这四份文件是 LLM 目录里**没有**的，它们定义了“骨干之后接什么”，是本模型的灵魂。

## 6. modules.json —— 推理流水线

``` json
[
  { "idx": 0, "name": "0", "path": "",          "type": "sentence_transformers.models.Transformer" },
  { "idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling" },
  { "idx": 2, "name": "2", "path": "2_Normalize","type": "sentence_transformers.models.Normalize" }
]
```

它定义了三级流水线，**顺序执行**：

``` text
① Transformer（就是 config.json 描述的骨干，输出 last_hidden_state）
        ↓
② Pooling    （按 1_Pooling/config.json 的方式池化 → 单个向量）
        ↓
③ Normalize  （L2 归一化 → 单位向量）
```

对推理引擎的意义：**你复用 LLM 的 Transformer 图作为 ①，再补上 ②③ 两个后处理算子即可。**

## 7. 1_Pooling/config.json —— 池化方式

``` json
{
  "word_embedding_dimension": 1024,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": true,      ← 采用 last-token pooling
  "include_prompt": true
}
```

- `pooling_mode_lasttoken: true`：取**最后一个 token** 的 hidden state 作为句向量（因果模型的标准做法）。
- `word_embedding_dimension: 1024`：输出维度 1024（支持 MRL，可截断到 32~1024 的任意维度）。
- `include_prompt: true`：Instruct 前缀那部分 token 也参与 pooling（不排除指令）。

## 8. 2_Normalize/ —— L2 归一化

该目录通常没有可配置参数（`Normalize` 模块无权重），作用是把 pooling 得到的向量除以其 L2 模长，
使 `‖v‖ = 1`。归一化后**余弦相似度 = 点积**，检索直接做内积即可。

## 9. config_sentence_transformers.json —— ST 级配置

``` json
{
  "prompts": {
    "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    "document": ""
  },
  "default_prompt_name": null,
  "similarity_fn_name": "cosine"
}
```

- `prompts.query`：**查询端**的 Instruct 指令前缀模板（可按任务/语言自定义，官方建议用英文）。
- `prompts.document`：**文档端**为空——文档不加任何指令。
- `similarity_fn_name: "cosine"`：官方约定用余弦相似度衡量向量距离。

---

# 三、README.md 要点速记

模型自带 README 里对推理最有价值的信息：

| 项目 | 值 |
| ---- | ---- |
| 模型类型 | Text Embedding |
| 支持语言 | 100+ 语言 |
| 参数量 | 0.6B |
| 上下文长度 | 32K |
| Embedding 维度 | 最高 1024，支持自定义 32~1024（MRL） |
| 层数 | 28 |
| 是否指令感知 | 是（query 建议加 Instruct） |
| 相似度 | 余弦 |

README 同时给了三种官方用法：`sentence-transformers`、`transformers`（含 `last_token_pool` 参考实现）、
`vLLM`，以及 TEI 部署命令，可作为对拍/验证的参考基准。

---

# 四、一张表看懂：LLM 目录 vs Embedding 目录

| 文件 | LLM (Qwen3-0.6B) | Embedding (Qwen3-Embedding-0.6B) | 说明 |
| ---- | ---- | ---- | ---- |
| config.json | ✅ | ✅（vocab/ctx/eos 略不同） | 骨干同构 |
| tokenizer.* / vocab / merges | ✅ | ✅ | 复用，末尾拼 `<|endoftext|>` |
| model.safetensors | ✅ ~1.5GB | ✅ ~1.19GB | 词表更小、无独立 lm_head |
| generation_config.json | ✅（采样默认值） | ⚠️ 存在但基本不用 | embedding 不生成 |
| configuration.json | ✅ | ✅ | 仓库管理用 |
| **modules.json** | ❌ | 🆕 | 定义 Transformer→Pooling→Normalize 流水线 |
| **1_Pooling/config.json** | ❌ | 🆕 | last-token pooling |
| **2_Normalize/** | ❌ | 🆕 | L2 归一化 |
| **config_sentence_transformers.json** | ❌ | 🆕 | Instruct 提示词 + 相似度函数 |

> 迁移结论：**加载权重、搭 Transformer 图的代码可与 LLM 共用；
> 只需按 `modules.json` 补上 Pooling + Normalize 两步，并处理好 Instruct 前缀与末尾 `<|endoftext|>`，
> 就能把你现有的 Qwen3-0.6B 推理框架扩展成 embedding 推理。**
