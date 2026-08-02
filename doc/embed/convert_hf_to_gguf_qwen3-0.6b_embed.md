# `convert_hf_to_gguf.py` 脚本解析（以 Qwen3-Embedding-0.6B 为例）

> 分析对象：[tools/convert_hf_to_gguf.py](../../tools/convert_hf_to_gguf.py)（llama.cpp 官方转换脚本，随
> [tools/gguf-py/](../../tools/gguf-py/) 包一起拷贝进本仓库）。
>
> 输入模型：`/home/lil72/data/models/Qwen3-Embedding-0.6B/`（HF safetensors，bf16）
> 输出产物：[models/qwen3-embedding-0.6b-f16.gguf](../../models/qwen3-embedding-0.6b-f16.gguf)（约 1.2GB，F16）
>
> 本文与 [../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)
> 对齐节序，只讲**与 llm 版 Qwen3-0.6B 的差异**，共通逻辑（`ModelBase.write()` 执行顺序、
> `Qwen3Model` 类继承、`GGUFWriter` 内存 KV 字典、张量按 safetensors 字典序物理排布等）请直接
> 参考 llm 版文档。
>
> 所有字段/顺序/张量列表均通过 `tools/gguf-py/gguf/scripts/gguf_dump.py` 对实际生成的
> gguf 文件回读验证。

---

## 1. 转换命令与整体流程

```bash
conda run -n llm python tools/convert_hf_to_gguf.py \
    /path/models/Qwen3-Embedding-0.6B \
    --outfile models/qwen3-embedding-0.6b-f16.gguf \
    --outtype f16
```

**与 llm 版流程完全一致**：`parse_args` → `load_hparams` → 通过 `architectures[0]="Qwen3ForCausalLM"`
找到 `Qwen3Model` 类 → 实例化 → `write()`。

> ⚠️ 注意 `architectures` 依然是 **`Qwen3ForCausalLM`**（而不是像 BERT 那种独立的 EmbeddingModel）。
> Qwen3-Embedding-0.6B 就是**基于 Qwen3-0.6B-Base 微调**得到的，模型结构与 Qwen3-0.6B 完全同构，
> 只是使用方式不同（取隐层做 pooling 而非产 logits 采样）。所以转换脚本走的分支和 llm 版**完全相同**。

### 1.1 类继承（复用 llm 版）
`Qwen3ForCausalLM` → `Qwen3Model`（继承 `Qwen2Model` → `TextModel` → `ModelBase`），
`model_arch = gguf.MODEL_ARCH.QWEN3`。详见 [llm 版 §1.1](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)。

### 1.2 `write()` 内部顺序（复用 llm 版）
`prepare_tensors()` → `prepare_metadata()` → `write_header_to_file` → `write_kv_data_to_file`
→ `write_tensors_to_file` → `close()`。详见 [llm 版 §1.2](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)。

---

## 2. 元数据（KV）写入：36 条（对比 llm 版 38 条）

用 `gguf_dump.py` 回读 [models/qwen3-embedding-0.6b-f16.gguf](../../models/qwen3-embedding-0.6b-f16.gguf) 得到：

| # | Key | 类型 | 值 | 与 llm 版对比 |
|---|-----|------|----|--------------|
| 1 | `general.architecture` | STRING | `qwen3` | ✅ 相同（走同一 `MODEL_ARCH.QWEN3`） |
| 2 | `general.type` | STRING | `model` | ✅ 相同 |
| 3 | `general.name` | STRING | **`Qwen3 Embedding 0.6B`** | ⚠️ 与 llm 版 `Qwen3 0___6B` 不同：HF 目录名是干净的 `Qwen3-Embedding-0.6B`，无 ModelScope 遗留 `___` 坑 |
| 4 | `general.basename` | STRING | **`Qwen3-Embedding`** | ⚠️ llm 版是 `Qwen3` |
| 5 | `general.size_label` | STRING | **`0.6B`** | ⚠️ llm 版是 `752M`（llm 版体积因独立 `output.weight` 副本膨胀，`size_label()` 按张量总参数量估算，落到不同区间） |
| 6 | `general.license` | STRING | `apache-2.0` | ✅ 相同 |
| 7 | `general.base_model.count` | UINT32 | `1` | ✅ |
| 8 | `general.base_model.0.name` | STRING | `Qwen3 0.6B Base` | ✅ 与 llm 版相同（Embedding 版基于同一 Base） |
| 9 | `general.base_model.0.organization` | STRING | `Qwen` | ✅ |
| 10 | `general.base_model.0.repo_url` | STRING | `https://huggingface.co/Qwen/Qwen3-0.6B-Base` | ✅ |
| 11 | `general.tags` | [STRING]×5 | `["transformers", "sentence-transformers", "sentence-similarity", "feature-extraction", "text-embeddings-inference"]` | ⚠️ llm 版是 `["text-generation"]`。tags 来自 HF README front-matter |
| 12 | `qwen3.block_count` | UINT32 | `28` | ✅ |
| 13 | `qwen3.context_length` | UINT32 | **`32768`** | ⚠️ llm 版是 `40960`（config.json 的 `max_position_embeddings` 不同：Embedding 版 32K 就够用） |
| 14 | `qwen3.embedding_length` | UINT32 | `1024` | ✅ |
| 15 | `qwen3.feed_forward_length` | UINT32 | `3072` | ✅ |
| 16 | `qwen3.attention.head_count` | UINT32 | `16` | ✅ |
| 17 | `qwen3.attention.head_count_kv` | UINT32 | `8` | ✅ |
| 18 | `qwen3.rope.freq_base` | FLOAT32 | `1000000.0` | ✅ |
| 19 | `qwen3.attention.layer_norm_rms_epsilon` | FLOAT32 | `1e-06` | ✅ |
| 20 | `qwen3.attention.key_length` | UINT32 | `128` | ✅ |
| 21 | `qwen3.attention.value_length` | UINT32 | `128` | ✅ |
| 22 | `general.file_type` | UINT32 | `1` (`MOSTLY_F16`) | ✅ |
| 23 | **`qwen3.pooling_type`** | UINT32 | **`3`** (`LAST`) | 🆕 llm 版**无此键**：`Qwen2Model.set_gguf_parameters()` 里的 `_try_set_pooling_type()` 会读 `1_Pooling/config.json`，Embedding 版发现 `pooling_mode_lasttoken=true` 于是写入 `3`（枚举 `NONE=0/MEAN=1/CLS=2/LAST=3/RANK=4`） |
| 24 | `general.quantization_version` | UINT32 | `2` | ✅ |
| 25 | `tokenizer.ggml.model` | STRING | `gpt2` | ✅（相同 BPE） |
| 26 | `tokenizer.ggml.pre` | STRING | `qwen2` | ✅ |
| 27 | `tokenizer.ggml.tokens` | [STRING]×**151669** | 词表 | ⚠️ llm 版是 `151936`：Embedding 版少了 **267** 个 token（llm 版为对话/工具追加了额外 special token） |
| 28 | `tokenizer.ggml.token_type` | [INT32]×151669 | 每个 token 的类型 | ⚠️ 长度同 #27 |
| 29 | `tokenizer.ggml.merges` | [STRING]×**151387** | BPE 合并规则 | ✅ 相同（merges.txt 与 llm 版共用 BPE 主体） |
| 30 | `tokenizer.ggml.eos_token_id` | UINT32 | **`151643`** (`<\|endoftext\|>`) | ⚠️ llm 版是 `151645` (`<\|im_end\|>`)：Embedding 版无 chat 场景，回落到 base eos |
| 31 | `tokenizer.ggml.padding_token_id` | UINT32 | `151643` | ✅（与 eos 共用） |
| 32 | `tokenizer.ggml.eot_token_id` | UINT32 | `151645` | ⚠️ llm 版此 key 不出现（脚本判定该模型无独立 eot） |
| 33 | `tokenizer.ggml.bos_token_id` | UINT32 | `151643` | ✅ |
| 34 | **`tokenizer.ggml.add_eos_token`** | BOOL | **`True`** | 🚨 llm 版**无此键**（默认 false）。**这是 Embedding 版最重要的分词差异**：官方 `AutoTokenizer` 会在每条文本末尾**自动追加 `<\|endoftext\|>`(151643)**，C++ 侧调用 `tokenizer.encode()` 时**不做**这个追加，必须手动 `ids.push_back(eos_id)`，否则 last-token pool 取到的不是 EOS 位置的 hidden，cosine 会对不上 |
| 35 | `tokenizer.ggml.add_bos_token` | BOOL | `false` | ✅ |
| 36 | `tokenizer.chat_template` | STRING | Jinja2 chat 模板全文 | ⚠️ Embedding 用途**不需要**它，但 tokenizer_config.json 里仍带着，脚本原样写入。frostfall.embed 侧根本不读这个键 |

---

## 3. 架构超参写入：只多了一个 `qwen3.pooling_type`

`Qwen3Model.set_gguf_parameters()` → `Qwen2Model.set_gguf_parameters()` → `TextModel.set_gguf_parameters()`
的调用链与 llm 版完全一致（[llm 版 §3](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)），
`config.json` 里能读到的通用字段一模一样：

| GGUF key | Qwen3-0.6B | Qwen3-Embedding-0.6B |
|---|---|---|
| `qwen3.block_count` | 28 | 28 |
| `qwen3.context_length` | **40960** | **32768** |
| `qwen3.embedding_length` | 1024 | 1024 |
| `qwen3.feed_forward_length` | 3072 | 3072 |
| `qwen3.attention.head_count` | 16 | 16 |
| `qwen3.attention.head_count_kv` | 8 | 8 |
| `qwen3.rope.freq_base` | 1000000.0 | 1000000.0 |
| `qwen3.attention.layer_norm_rms_epsilon` | 1e-6 | 1e-6 |
| `qwen3.attention.key_length` / `value_length` | 128 / 128 | 128 / 128 |

**唯一多出的字段**：`qwen3.pooling_type = 3`，由 `Qwen2Model.set_gguf_parameters()` 里的
`self._try_set_pooling_type()` 触发。该辅助函数会检查模型目录里是否存在
[`1_Pooling/config.json`](../../.././../data/models/Qwen3-Embedding-0.6B/1_Pooling/config.json)，
读取 `pooling_mode_*` 字段判定 pooling 方式：

```json
{
    "pooling_mode_cls_token":         false,
    "pooling_mode_mean_tokens":       false,
    "pooling_mode_max_tokens":        false,
    "pooling_mode_mean_sqrt_len_tokens": false,
    "pooling_mode_weightedmean_tokens":  false,
    "pooling_mode_lasttoken":         true,      ← 命中，映射到 gguf.PoolingType.LAST = 3
    "include_prompt":                 true
}
```

llm 版目录里没有这个文件，`_try_set_pooling_type()` 直接跳过，因此 KV 里就没有 `pooling_type` 字段。

> **注意**：本项目 [src/embed/embed_graph.cpp](../../src/embed/embed_graph.cpp) 并未读取 `qwen3.pooling_type`，
> 而是**在 CPU 侧硬编码 last-token pool**（[src/embed/pooling.cpp](../../src/embed/pooling.cpp)）。因为 Qwen3-Embedding
> 系列（0.6B/4B/8B）的 pooling 方式官方全部锁死为 LAST，为一个 Embedding 模型加运行时分派没必要；
> 后续如果需要支持 MEAN pooling 的其它模型再增加分支。

Qwen3 特有的两个张量级细节（QK-Norm、`head_dim=128 ≠ hidden_size/head_count`）与 llm 版完全相同，
详见 [llm 版 §3 末尾](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)。

---

## 4. 分词器写入：走同一 `_set_vocab_gpt2()` 分支，但两处差异要留意

调用链与 llm 版一致：`Qwen3Model.set_vocab()` → 不命中 `intern-s1-mini` → `Qwen2Model.set_vocab()`
→ `_set_vocab_sentencepiece` 抛 `FileNotFoundError`（无 `tokenizer.model`）→ **`_set_vocab_gpt2()`**。

### 4.1 词表少 267 个 token（`151669` vs `151936`）

Qwen3-Embedding-0.6B 的 `tokenizer.json` 里 `added_vocab` 精简掉了 llm 版为
对话/工具场景准备的额外 special token（`<|box_start|>`、`<|object_ref_start|>` 等约 260 多个
未在 Embedding 场景使用的 token）。`_set_vocab_gpt2()` 会照实反映在词表里：
- llm 版：`vocab_size=151936`（config.json 里显式声明），实际词表 `151936`；
- Embedding 版：`vocab_size=151669`（config.json 里显式声明），实际词表 `151669`。

C++ 侧 `n_vocab` 是**从 `token_embd.weight` 的 `ne[1]` 读的**（见
[src/core/model.cpp](../../src/core/model.cpp) 里 `hp.n_vocab = model.tok_embd->ne[1]`），不是从 KV 里读，
所以这个差异对推理没有额外影响。

### 4.2 `tokenizer.ggml.add_eos_token = True`（llm 版是隐式 false）

这是**本转换脚本对 Embedding 场景唯一一个"看不见的坑"**：
- Embedding 版 `tokenizer_config.json` 里 `add_eos_token: true`；
- `gguf.SpecialVocab` 会把该布尔字段写进 GGUF；
- 但 [src/core/tokenizer.cpp](../../src/core/tokenizer.cpp) 目前**只解析 `add_bos_token`，忽略 `add_eos_token`**
  （llm 版一直用 `false` 所以从未处理过）。

**后果**：如果直接在 C++ 里用 `tokenizer.encode(text)`，得到的 ids 序列末尾**不会**带 EOS；
而官方 `AutoTokenizer.__call__(...)` 会带。Embedding 采用 last-token pool，序列末位的 hidden
决定输出向量，因此少了这个 EOS 会导致 cosine 与官方显著偏差。

**修复**：在调用层（当前 [examples/embed/v0_1_smoke/v0_1_smoke.cpp](../../examples/embed/v0_1_smoke/v0_1_smoke.cpp)）显式追加：

```cpp
std::vector<int32_t> ids = tokenizer.encode(input);
int32_t eos_id = model.hparams.eos_token_id > 0 ? model.hparams.eos_token_id : 151643;
ids.push_back(eos_id);
```

v1.0 引入 `EmbeddingEngine` 时会把这段逻辑收进引擎，由引擎读 `hparams.eos_token_id`
+ `add_eos_token` 标志决定是否自动追加，不再暴露给用户。

---

## 5. 张量写入：310 个（对比 llm 版 311）

### 5.1 处理管线：与 llm 版一致
`prepare_tensors()` → `modify_tensors()` → 权重是 `_norm.weight` 结尾或 `n_dims<=1` 强制 F32、
其余走 F16 → `gguf_writer.add_tensor(...)`。详见 [llm 版 §5.1](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)。

### 5.2 单层张量：完全相同（11 个）

Qwen3-Embedding-0.6B 与 Qwen3-0.6B 每层的 11 个张量**名字、形状、dtype 全部一致**
（`attn_norm` / `attn_q,k,v,output` / `attn_q_norm,k_norm` / `ffn_norm,gate,up,down`），
详见 [llm 版 §5.2 单层张量表](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)。

### 5.3 全局张量：**少了 `output.weight`**

| GGUF 张量名 | 形状 | dtype | llm 版 | Embedding 版 |
|---|---|---|---|---|
| `token_embd.weight` | `[1024, n_vocab]` | F16 | ✅ 有 | ✅ 有 |
| `output_norm.weight` | `[1024]` | F32 | ✅ 有 | ✅ 有 |
| `output.weight` | `[1024, n_vocab]` | F16 | ✅ 有（**多余一份**，与 embedding 数值相同） | **❌ 无** |

- **总张量数**：llm 版 `28×11 + 3 = 311`，Embedding 版 `28×11 + 2 = 310`。
- **文件体积**：llm 版 `qwen3-0.6b-f16.gguf ≈ 1.4 GB`（含冗余 lm_head 副本约 300MB），
  Embedding 版 `qwen3-embedding-0.6b-f16.gguf ≈ 1.2 GB`。

**原因**：两个模型的 `config.json` 都写着 `tie_word_embeddings=true`，但差别在 safetensors 存储：
- llm 版 `Qwen3-0.6B/model.safetensors` **显式保存了 `lm_head.weight`**（虽然与 embed_tokens 数值相同），
  转换脚本原样写入产生 `output.weight`。
- Embedding 版 `Qwen3-Embedding-0.6B/model.safetensors` **没有独立的 `lm_head.weight`**，
  转换脚本自然也不会写入 `output.weight`。

**推理端如何应对**：[src/core/model.cpp](../../src/core/model.cpp) 在加载时已经有 fallback 逻辑：

```cpp
model.output = ggml_get_tensor(model.ctx_data, "output.weight");
if (!model.output) {
    model.output = model.tok_embd;   // 回落到 tied embedding
    LOG(INFO) << "'output.weight' not found, reusing token_embd (tied embedding)";
}
```

**embed 版根本不用 `model.output`**（[src/embed/embed_graph.cpp](../../src/embed/embed_graph.cpp) 只到 final RMSNorm
就截止，没有 lm_head 矩阵乘），所以这个 fallback 只是"不影响加载"，实际零内存开销。

### 5.4 物理顺序（复用 llm 版结论）

张量按 **safetensors 内部 key 字典序** 物理排布，因此层号是 `0, 1, 10, 11, ..., 19, 2, 20, ..., 9`
而非数值 `0..27`。Embedding 版首个张量是 **`token_embd.weight`**（llm 版首个是 `output.weight`，
因为 `lm_head < model.embed_tokens` 字典序），末尾都是 `output_norm.weight`。
详见 [llm 版 §5.3](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)。

---

## 6. 文件物理布局（复用 llm 版）

四段结构完全一致：`Header → KV 数据 → 张量信息表 → 张量二进制数据`，只是数值不同：
- KV 数：36（llm 38）
- 张量数：310（llm 311）
- 词表条目数：151669（llm 151936）
- 总大小：~1.2 GB（llm ~1.4 GB）

结构图直接参考 [llm 版 §6](../llm/convert_hf_to_gguf_qwen3-0.6b_llm.md)。

---

## 7. 小结：Embedding vs LLM 的 GGUF 差异一览

| 类别 | 差异点 | 影响 |
|---|---|---|
| **架构** | 无（`architectures=Qwen3ForCausalLM`，走同一 `Qwen3Model` / `MODEL_ARCH.QWEN3`） | ✅ 复用 llm 侧模型加载与 28 层 block 前向逻辑 |
| **KV 新增** | `qwen3.pooling_type = 3 (LAST)` | 记录官方 pooling 意图；本项目在 [src/embed/pooling.cpp](../../src/embed/pooling.cpp) 硬编码 last-token pool，不读该字段 |
| **KV 新增** | `tokenizer.ggml.add_eos_token = True` | 🚨 **必须**在 encode 后手动追加 `<\|endoftext\|>`(151643)，否则 last-token pool 位置错、cosine 掉 |
| **KV 变化** | `context_length`: 40960 → 32768 | Embedding 32K 足够，无需扩窗 |
| **KV 变化** | `eos_token_id`: 151645 → 151643（回落到 `<\|endoftext\|>`） | Embedding 无 chat 场景，用 base 版 eos |
| **KV 元信息** | `name / basename / size_label / tags` 都不同 | 只用于展示，不影响推理 |
| **词表** | 151936 → 151669（少 267 个 chat 用途 special token） | `n_vocab` 从 tensor 形状读，自适应，无需改代码 |
| **张量** | 无独立 `output.weight`（310 个 vs 311 个） | 推理端 `model.output` fallback 到 `tok_embd`，embed 图根本不用 lm_head，零成本 |
| **张量物理顺序** | 首个是 `token_embd.weight`（llm 版是 `output.weight`） | 按名字读取，与顺序无关 |
| **`tokenizer.chat_template`** | 仍写入（脚本原样带上） | Embedding 不读，无害 |

**一句话总结**：Qwen3-Embedding-0.6B 的 GGUF 与 Qwen3-0.6B 结构上**完全同构**，转换脚本走同一路径，
产物差异集中在词表大小、几个元信息字段、一个 `pooling_type` 新增键、`add_eos_token` 标志翻转，
以及少了一份冗余的 `output.weight` 副本。真正对推理**行为**有影响的只有一处：
`add_eos_token=True` 必须在调用层显式模拟，否则 last-token pool 取位错误。
