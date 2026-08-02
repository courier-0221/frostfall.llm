# frostfall.embed —— 基于 ggml 的 Qwen3-Embedding-0.6B 推理框架设计方案

> 在 `frostfall.llm`（Qwen3-0.6B 因果 LM 推理）v0.1 → v1.0 已跑通的基础上，
> 本方案复用其绝大部分基础设施（GGUF 加载、Qwen3 权重组织、单层前向、tokenizer、CMake、
> ggml/nlohmann 依赖），仅替换 **"生成 logits + 采样自回归"** 为 **"取最后隐层 + pooling + L2 归一化"**，
> 落地 Qwen3-Embedding-0.6B 的文本向量化推理。
>
> 目标模型：`Qwen3-Embedding-0.6B`（GGUF，F16），只跑 CPU backend。
> 参考模型路径：`/home/lil72/data/models/Qwen3-Embedding-0.6B/`。

---

## 1. 项目定位与设计原则

### 1.1 目标
- 只做 **embedding 推理**：文本 → 定长向量（默认 1024 维）。
- 只支持 **一个模型**：Qwen3-Embedding-0.6B。
- 只跑 **CPU backend**（ggml v0.15.3）。
- 最大限度**复用 `frostfall.llm` 的代码**（详见 §4），主体新代码控制在 **500 行 C++** 以内。

### 1.2 刻意不做的事
| 场景 | 取舍 |
| --- | --- |
| 训练 / 微调 | 不做，只推理 |
| Reranker（Qwen3-Reranker） | 不做，Reranker 走生成式打分（是另一条推理路径），可留作后续 `frostfall.rerank` |
| Sparse embedding / ColBERT 风格 | 不做 |
| 多 backend（CUDA/Metal） | 不做，只 CPU |
| GRPC / HTTP server | 不做，只出共享库 + 示例 CLI |
| 动态 batch / continuous batching | 不做，v1.0 支持"一次调用一次 batch"的静态 batch |

### 1.3 设计原则
1. **可读性 > 性能 > 通用性**，与 `frostfall.llm` 一致。
2. **只分 2 个版本**（v0.1 主链路 + v1.0 库化收口）——
   Embedding 推理链路本身很薄，LLM 版 v0.2/v0.3 的独立里程碑（自研分词、增量 KV、采样）
   在此都不存在，硬拆多版反而稀释每一版的价值。
3. **对拍先行**：v0.1 就要能与 `transformers` 官方输出做 cosine 对拍。
4. **接口与 `frostfall.llm` 对齐但不复用**：`EmbeddingEngine` 类比 `InferenceEngine`，
   风格一致（JSON 初始化、`namespace Frostfall`、线程安全、无状态调用），但独立成类。

---

## 2. Qwen3-Embedding-0.6B 模型速览

### 2.1 与 Qwen3-0.6B（LLM 版）的超参数对比
读自 `/home/lil72/data/models/Qwen3-Embedding-0.6B/config.json`：

| 参数 | LLM 版 Qwen3-0.6B | Embedding 版 Qwen3-Embedding-0.6B | 是否一致 |
| --- | --- | --- | --- |
| `architectures` | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` | ✅ 一致（Embedding 版基于 Qwen3-0.6B-Base，架构完全相同） |
| `hidden_size` | 1024 | 1024 | ✅ |
| `num_hidden_layers` | 28 | 28 | ✅ |
| `num_attention_heads` | 16 | 16 | ✅ |
| `num_key_value_heads` | 8 | 8 | ✅（GQA 8） |
| `head_dim` | 128 | 128 | ✅ |
| `intermediate_size` | 3072 | 3072 | ✅ |
| `rms_norm_eps` | 1e-6 | 1e-6 | ✅ |
| `rope_theta` | 1e6 | 1e6 | ✅ |
| `max_position_embeddings` | 40960 | 32768 | ⚠️ 略不同（32K 就够用） |
| `tie_word_embeddings` | true | true | ✅（但 embedding 版根本不用 `lm_head`） |
| `vocab_size` | 151936 | **151669** | ⚠️ 词表略缩（少了部分未使用的保留 token，BPE 主体相同） |

> **结论**：Embedding 版的 Transformer 主体（28 层 GQA + QK-Norm + SwiGLU + RMSNorm + RoPE-NEOX）
> 与 LLM 版**完全同构**，`src/core/model.{h,cpp}` / `src/llm/graph.{h,cpp}` 的层内实现（RMSNorm/RoPE/GQA/SwiGLU）
> 可原样复用，只是 `n_vocab` 与 `n_ctx_train` 的数值不同，且**不再需要 `output.weight`（lm_head）**。

### 2.2 Pooling / 归一化配置
读自 `Qwen3-Embedding-0.6B/modules.json` + `1_Pooling/config.json`：

```
Transformer  →  Pooling(last_token=true, include_prompt=true)  →  Normalize(L2)
```

- **Pooling 方式**：`pooling_mode_lasttoken=true`（其它 5 种均 false）。
  取最后一层 hidden states 里 **最后一个非 pad token** 位置的向量。
- **左 padding**：官方 `transformers` 用法是 `padding_side='left'`；
  在左 padding 下，最后一个位置一定是真实 token，所以直接取 `hidden[:, -1, :]` 即可（对齐官方 `last_token_pool`）。
- **归一化**：L2 normalize（`F.normalize(embeddings, p=2, dim=1)`），使 `‖v‖₂ = 1`，
  下游用点积即可得 cosine 相似度。
- **输出维度**：默认 1024；模型支持 **MRL（Matryoshka Representation Learning）**，
  可在 [32, 1024] 内截断前 N 维再重新归一化，得到 N 维向量。

### 2.3 Query 侧 Instruction 模板
读自 `config_sentence_transformers.json`：

```
Instruct: {task_description}\nQuery:{query}
```

- **query 侧**：在文本前拼接上述模板（`task_description` 可由业务传入，默认
  `"Given a web search query, retrieve relevant passages that answer the query"`）。
- **document 侧**：**不加**任何前缀，原文直接送进 tokenizer。
- 注意与 LLM 版 ChatML 完全不同：**没有 `<|im_start|>` / `<|im_end|>` 包裹，没有 role**。

### 2.4 整体前向伪代码
```
text                                                # 输入文本
  └─ maybe_prepend_instruction(text, is_query, task)  # query 侧才拼 Instruct 模板
        └─ tokenizer.encode(text) ──► ids [n]        # 复用 LLM 版 tokenizer
              └─ build_graph(ids, positions[0..n-1]):
                   x = get_rows(token_embd, ids)     # [n_embd, n]
                   for l in 0..n_layer-1:            # 与 LLM 版单层完全一致
                        x = block_forward(x, l, mask=causal)
                   x = rms_norm(x) * output_norm     # 最后一次 final RMSNorm
                   ── 不做 lm_head 投影 ──
              └─ pooling: v = x[:, -1]               # 取最后位置(左padding下=最后一个真实token)
              └─ (可选 MRL) v = v[:target_dim]
              └─ normalize: v = v / ‖v‖₂
                     └─ float[target_dim]            # 输出向量
```

> **注意**：Embedding 模型仍是 **causal（因果）** 注意力（继承自 Qwen3-0.6B-Base，非 BERT 那种双向）。
> last-token pool 的语义就是"最后一个位置能看见前面全部 token"，因此
> `soft_max_ext` 的 causal mask **必须保留**，不能改成全 0 的双向 mask。

---

## 3. 与 LLM 版推理链路的核心差异

| 环节 | LLM 版（`frostfall.llm`） | Embedding 版（`frostfall.embed`） |
| --- | --- | --- |
| Chat 模板 | Qwen3 ChatML（`<|im_start|>role\n...<|im_end|>`） | ❌ 不用；query 侧拼 `Instruct:\nQuery:` 前缀 |
| Tokenizer | 复用 `src/core/tokenizer.{h,cpp}`（BPE + special tokens） | ✅ 完全复用，无改动 |
| KV cache | 增量 KV cache（v0.2 起），prefill + decode | ❌ **不需要**：单次前向、无自回归、无 n_past 概念 |
| 计算图 | build 一次前向图 → 取最后位置 logits | build 一次前向图 → 取最后位置 hidden（**不过 lm_head**） |
| `lm_head` / `output.weight` | 用（tied 到 token_embd） | ❌ 不用（连读都不用读） |
| 采样 | greedy / temp / top-k / top-p / repeat penalty | ❌ 不用 |
| 生成循环 | prefill → 逐 token decode 至 EOS | ❌ 一次前向就结束 |
| EOS / stop | `<|im_end|>` / `<|endoftext|>` | ❌ 不涉及 |
| 输出 | `LlmResponse{content, thinking, tool_calls}` | `EmbedResponse{vector<float>, dim, n_tokens}` |
| Pooling | ❌ | ✅ last-token pool（左 padding 下 = 序列末位） |
| L2 归一化 | ❌ | ✅ 必做 |
| MRL 维度截断 | ❌ | ✅ 可选（32..1024） |
| Batch | ❌（v1.0 也是单序列） | ✅ v1.0 支持（左 padding + attention mask） |
| Thinking / tool_call 拆分 | ✅ | ❌ 不涉及 |

**一句话总结**：把 LLM 版的 `graph.cpp` **砍掉最后的 `lm_head` 矩阵乘 + KV cache 读写**，
把 `main` 的"prefill + decode 循环 + 采样"整块换成 **"一次 compute → 取最后位置 hidden → 归一化"**，
其它全部照抄。

---

## 4. 复用与新增模块清单

### 4.1 直接复用（零改动或极小改动）
- `src/core/tokenizer.{h,cpp}` —— **零改动**。BPE + special tokens 加载与 encode/decode 与 LLM 版完全一致。
- `src/core/common.{h,cpp}` —— 零改动。GGUF hparams 读取、日志、计时。
- `src/core/log.h` —— 零改动。
- `third_party/ggml/` —— 零改动。
- `third_party/nlohmann/json.hpp` —— 零改动（v1.0 JSON 配置化用）。
- `tools/convert_hf_to_gguf.py` —— **零改动**。该脚本已能识别 `Qwen3ForCausalLM`（走 `Qwen3Model`），
  Qwen3-Embedding-0.6B 的架构相同即可原样转换；产物: `models/qwen3-embedding-0.6b-f16.gguf`。
- CMake 主结构、模块源文件列表的组织模式 —— 复用；embed 源文件以 `FROSTFALL_EMBED_SOURCES`
  加入**同一个**共享库 `libfrostfall.so`（决策见 §4.4）。

### 4.2 需要裁剪 / 特化的模块
- `src/core/model.{h,cpp}` —— **v0.1 不动**（允许 `output.weight` 白加载 300MB），
  **v1.0 加 `skip_output` 参数**：Qwen3-Embedding-0.6B 由于 `tie_word_embeddings=true`，
  转换脚本仍会写一份 `output.weight`，推理端跳过后可节省 ~300MB 内存。张量命名、hparams KV
  读取逻辑本身完全复用。
- `src/embed/embed_graph.{h,cpp}` —— **新增**（**不改** `src/llm/graph.{h,cpp}`，llm 图独立保留）。
  照抄 `graph.cpp` 骨架，做 3 处砍减：
  - 保留 embed / 28 层 block / final RMSNorm。
  - **删掉 lm_head 矩阵乘**，图的最终输出改为 `[n_embd, n_tokens]` 的最后隐层张量。
  - **删掉写 KV cache 的 `ggml_cpy` 节点**，K/V 只在本次前向内使用（局部张量，随图释放）。
  - `soft_max_ext` 仍用 causal mask（不改成双向）。

### 4.3 全新新增
| 文件 | 引入时机 | 职责 |
| --- | --- | --- |
| `src/embed/embed_graph.{h,cpp}` | v0.1 | embed 专用计算图（照抄 `graph.cpp` 骨架，砍 KV cpy + lm_head，输出改为最后隐层） |
| `src/embed/pooling.{h,cpp}` | v0.1 | last-token pool + MRL 截断 + L2 normalize |
| `src/embed/qwen3_embed_prompt.{h,cpp}` | v0.1 | query instruction 模板拼接（`Instruct:\nQuery:` 前缀） |
| `src/embed/embedding_engine.{h,cpp}` | v1.0 | 引擎门面：`Init(EngineConfig)` / `Embed(text)` / `Embed(batch)` / `GetLastStats()` |
| `src/embed/embed_types.h` | v1.0 | `EmbedRequest` / `EmbedResponse` / `EmbedStats`（无 sampling、无 stop、无 finish_reason） |
| `examples/embed/*` | v0.1/v1.0 | v0.1 一个 smoke；v1.0 两个示例 CLI（blocking + batch）+ 共享 utils |
| `scripts/embed_reference.py` | v0.1 | 用 `transformers` 生成参考向量，供 C++ 侧对拍 |

### 4.4 起步基线：从 llm v1.0 的 `src/` 树出发

Embedding 不基于空白实现，也不从 llm 的某个中间版本起步 —— **直接以 llm v1.0 收口后
的 `src/` 树作为起点**。

**为什么必须是 v1.0**：

| 依赖项 | llm 侧引入时机 | Embed v0.1 是否需要 |
| --- | --- | --- |
| `src/core/tokenizer.{h,cpp}`（自研 BPE） | **v0.2 才有** | ✅ 必须（v0.1 是 Python 兜底，无法用） |
| `src/core/common.{h,cpp}` 已去掉 `namespace ff` | **v1.0 M1 完成** | ✅ 直接用更干净、无冲突 |
| CMake 模块源文件列表 + 共享库组织 | **v1.0 M5 完成** | ✅ Embed 直接照抄这套骨架 |
| `src/core/model.{h,cpp}` / `src/llm/graph.{h,cpp}` | v0.1 起就有，稳定 | ✅ 直接复用 |
| `kv_cache` / `sampler` / `inference_engine` / `qwen3_chat` | v0.2/v0.3/v1.0 陆续加入 | ❌ 不 include、不链接（源码留着零成本共存） |

回退到 v0.2 或 v0.3 都能跑，但要么少了"去 `namespace ff`"的干净收口，要么少了 v1.0
的静态库 CMake 骨架 —— llm v1.0 新增的那些"embed 用不到"的模块，对 embed 是**纯粹的零成本共存**。

**Embed v0.1 阶段实际动多少代码**（在 llm v1.0 现有 `src/` 之上）：

1. **零改动直接 include/link**：`core/tokenizer.{h,cpp}` / `core/common.{h,cpp}` / `core/log.h` /
   `core/model.{h,cpp}` / `third_party/ggml/` / `third_party/nlohmann/json.hpp` / `tools/convert_hf_to_gguf.py`。
2. **新增 4 个 embed 专属文件**（不污染 llm 侧）：
   - `src/embed/embed_graph.{h,cpp}` —— embed 专用计算图（**照抄 `graph.cpp` 骨架**，砍掉 KV cpy + lm_head，输出改成最后一层 hidden）。
   - `src/embed/pooling.{h,cpp}` —— last-token pool + MRL 截断 + L2 归一化（几十行）。
   - `src/embed/qwen3_embed_prompt.{h,cpp}` —— `Instruct:\nQuery:` 前缀拼接（十几行）。
   - `examples/embed/v0_1_smoke/v0_1_smoke.cpp` —— 函数式 API 的最小 CLI。
   - `scripts/embed_reference.py` —— Python 参考实现（不在 src/，此处顺带列出）。
3. **不动 / 不新增**：
   - ❌ **不改 `src/llm/graph.{h,cpp}`**（llm 的图独立保留，走 `embed_graph.cpp` 新写一份，教学价值高）。
   - ❌ **不改 `src/core/model.{h,cpp}`**（v0.1 允许 `output.weight` 白加载 300MB；v1.0 再加 `skip_output` 参数优化）。
   - ❌ 不新增 `embedding_engine.*` / `embed_types.h`（留给 v1.0）。

**CMake 组织（定稿：单一共享库 + src 模块目录）**：

> **决策记录**：初版方案是"v0.1 把 embed 源文件直接编进 smoke 可执行文件、v1.0 再抽独立静态库
> `libfrostfall_embed`"。定稿改为：**v0.1 起 embed 源文件就收进统一的 `libfrostfall.so`**，
> 同时把 `src/` 按任务划分为 `core / llm / embed` 三个模块目录。理由：
> 1. examples 只应依赖三条契约——target 名 `frostfall`、`PUBLIC` include 目录、扁平头文件包含方式；
>    直接编译 `${CMAKE_SOURCE_DIR}/src/*.cpp` 会把 example 耦合到 src 物理路径，目录一搬就断。
> 2. 目录划分（可读性）与库产物数量（部署）是两件事，教学规模下单库最简单；
>    真出现"只部署 embed"诉求时再拆库，且拆库同样不破三条契约，examples 仍无感。
> 3. `graph.cpp`（llm）与 `embed_graph.cpp`（embed）保持分离、不做公共层抽取：两条前向链路各自成文，
>    教学叙事更清晰。

```cmake
# 根 CMakeLists.txt：模块源文件列表 + 单一共享库
set(FROSTFALL_CORE_SOURCES   src/core/model.cpp src/core/tokenizer.cpp src/core/common.cpp)
set(FROSTFALL_LLM_SOURCES    src/llm/graph.cpp src/llm/kv_cache.cpp src/llm/sampler.cpp
                             src/llm/qwen3_chat.cpp src/llm/inference_engine.cpp)
set(FROSTFALL_EMBED_SOURCES  src/embed/embed_graph.cpp src/embed/pooling.cpp
                             src/embed/qwen3_embed_prompt.cpp)   # v1.0 追加 embedding_engine.cpp

add_library(frostfall SHARED ${FROSTFALL_CORE_SOURCES} ${FROSTFALL_LLM_SOURCES} ${FROSTFALL_EMBED_SOURCES})
target_include_directories(frostfall PUBLIC src/core src/llm src/embed third_party)

# examples/embed/v0_1_smoke/CMakeLists.txt：薄壳，只链接 frostfall
add_executable(v0_1_smoke v0_1_smoke.cpp)
target_link_libraries(v0_1_smoke PRIVATE frostfall)
```

v1.0 不需要任何 CMake 重构：把 `embedding_engine.cpp` 加进 `FROSTFALL_EMBED_SOURCES`、
新增 `examples/embed/` 下两个示例目录即可（详见 §6.3）。

---

## 5. 版本路线图

只分 **2 个版本**。每一版都可编译、可运行、可与官方对拍。

| 版本 | 主题 | 产出 / 验收标准 |
| --- | --- | --- |
| **v0.1** | 单条推理主链路（含 instruction 模板 + MRL） | 单条文本（可选 query instruction 前缀）→ 1024 维（或 MRL 截断到 target_dim）向量；与 `transformers` 官方 `last_token_pool + F.normalize` 逐值对拍 **cosine ≥ 0.999**；官方 README 2×2 示例的相似度矩阵每元素与 `[[0.7646, 0.1414], [0.1355, 0.6000]]` **绝对误差 ≤ 0.01**。 |
| **v1.0** | 库化 + 批处理（通用接口收口） | 不新增库产物：`EmbeddingEngine` 收进统一共享库 `libfrostfall.so`（JSON 三入口、线程安全、无状态）；一次调用支持多条文本（左 padding + attention mask）；提供两个示例 CLI（单条 blocking + batch），覆盖 query/document/batch/MRL/长文本/多线程。 |

> **为什么不分 v0.2/v0.3？** 对齐到 LLM 版时它们的独立价值不成立：
> LLM v0.2 是"自研分词 + 增量 KV cache"，Embedding 分词完全复用、根本无 KV；
> LLM v0.3 是采样策略，Embedding 无采样。剩下的 batch/padding、instruction 模板、MRL
> 都是几十行的小改动，与主链路（v0.1）或库化（v1.0）自然合并即可，硬拆反而稀释每版的可验收产出。
>
> **无量化/CUDA 支线**：与 LLM 版 v0.4 类似的进阶优化正交、不阻塞，本方案不排期。

---

## 6. 各版本详细任务

### v0.1 —— 单条推理主链路（含 instruction 模板 + MRL）

**目标**：把"加载 GGUF → 构图（不含 lm_head）→ 一次前向 → 取最后位置 → 可选 MRL 截断 → L2 归一化 → 输出向量"
的最短闭环跑通，并建立与 `transformers` 官方的对拍手段。函数式 API（不做类封装），
命令行读文本、打印向量。

**任务拆解**：
1. **GGUF 转换**：`conda run -n llm python tools/convert_hf_to_gguf.py \
   /home/lil72/data/models/Qwen3-Embedding-0.6B --outtype f16 \
   --outfile models/qwen3-embedding-0.6b-f16.gguf`。
2. **模型加载**（`src/core/model.{h,cpp}`）：复用 LLM 版加载器，跳过 `output.weight` 张量。
   校验 hparams：`n_embd=1024, n_layer=28, n_head=16, n_head_kv=8, head_dim=128, vocab=151669`。
3. **计算图**（`src/embed/embed_graph.{h,cpp}`，`src/llm/graph.{h,cpp}` 的特化版）：
   - 输入：`ids [n_tokens]`、`positions [n_tokens]`。
   - 复用 28 层 block（RMSNorm → Q/K/V + QK-Norm → RoPE-NEOX → GQA 因果注意力 → SwiGLU FFN + 残差）。
   - final RMSNorm 后**直接输出 `[n_embd, n_tokens]`**（去掉 lm_head 矩阵乘）。
   - **删掉写 cache_k/cache_v 的 `ggml_cpy` 节点**：K/V 走本次前向内的临时张量即可。
4. **Instruction 模板**（`src/embed/qwen3_embed_prompt.{h,cpp}`）：
   ```cpp
   std::string build_query_text(const std::string& text,
                                const std::string& task_desc);
   // 返回: "Instruct: " + task_desc + "\nQuery:" + text
   ```
   `task_desc` 默认值取自 `config_sentence_transformers.json` 的
   `"Given a web search query, retrieve relevant passages that answer the query"`。
   `is_query=false`（document 侧）不做任何拼接。
5. **Pooling + MRL 截断 + 归一化**（`src/embed/pooling.{h,cpp}`）：
   - 从最后一层输出取 **最后一列**（`ne[1]-1` 位置）得到 `[n_embd]`。
   - MRL：`target_dim ∈ [32, 1024]`，**先截断前 target_dim 维**。
   - L2 归一化：`v[i] /= sqrt(sum(v[i]*v[i]))`。
   - **顺序不能反**：先归一化再截断会失去单位向量性质。
6. **CLI 或最小示例**：`examples/embed/v0_1_smoke/v0_1_smoke.cpp` 读一段文本，接受
   `--is-query` / `--task` / `--target-dim` 参数，打印前 8 维数值 + `‖v‖₂`。
7. **对拍脚本** `scripts/embed_reference.py`：
   ```python
   # conda run -n llm python scripts/embed_reference.py "The capital of China is Beijing."
   from transformers import AutoTokenizer, AutoModel
   import torch, torch.nn.functional as F
   tok = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
   mdl = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16)
   ids = tok(text, return_tensors='pt')
   h = mdl(**ids).last_hidden_state         # [1, n, 1024]
   v = h[0, -1]                             # last-token pool (无 padding 时等价)
   v = F.normalize(v, p=2, dim=0)
   ```
   把 v 前 8 维和 `‖v‖₂` 写进文本文件，C++ 侧读文件做 diff。

**验收**：
- 单条文本（含/不含 instruction）与官方对拍 **cosine ≥ 0.999**，前 8 维绝对误差 ≤ 5e-3（F16 累计误差）。
- 官方 README 2×2 示例（2 query + 2 doc）逐条单跑，相似度矩阵每元素与
  `[[0.7646, 0.1414], [0.1355, 0.6000]]` **绝对误差 ≤ 0.01**（v0.1 无 batch，2×2 是 4 次单条调用）。
- MRL：`target_dim ∈ {256, 512, 1024}` 均能返回单位向量（`‖v‖₂ ≈ 1`），高维检索排名与官方一致。

**允许偷懒**：
- 只支持**单条文本**，不 padding、不 batch（batch 留给 v1.0）。
- 函数式 API，不做类封装（收口留给 v1.0）。
- 无 JSON 配置文件（v1.0 引入）。

---

### v1.0 —— 库化 + 批处理（通用接口封装）

**目标**：对齐 `frostfall.llm` v1.0，把 `frostfall.embed` 收口进统一共享库 `libfrostfall.so` 供业务集成；
同时补上唯一一个还没做的算法特性 —— **静态 batch（左 padding + attention mask）**。

**v1.0 只干 3 件事**（区别于 LLM v1.0 M1..M5 的大工作量）：
1. **把 v0.1 的裸函数封装成 `EmbeddingEngine` 类**（`std::unique_ptr<Impl>` + `std::mutex` + JSON 三入口）。
2. **加 batch 能力**（左 padding + attention mask + graph 扩一个 batch 维）—— 唯一"新算法"。
3. **提供 2 个测试 CLI + 配置 JSON**（blocking 单条 + batch，覆盖 §6.3 全部用例）。

#### 6.1 静态 batch（v1.0 唯一新算法特性）

1. **左 padding**：以 batch 内最长序列为 `max_len`，短序列左侧补 `pad_id`
   （Qwen 系用 `<|endoftext|>` = 151643 作 pad；从 GGUF `tokenizer.ggml.padding_token_id` 或
   `tokenizer.ggml.eos_token_id` 拿）。这样每条序列的**最后一个位置** `[max_len-1]` 就是真实 token，
   直接取 `hidden[b, -1, :]` 就是 last-token pool 结果，与 v0.1 单条路径**共用同一段 pool 代码**。
2. **attention mask**：`ggml_soft_max_ext` 支持外部 mask（`float [n_kv, n_tokens]`）。
   构造 mask：对 pad 位置的 Key 列填 `-INFINITY`，其它按 causal 规则填（下三角 0，上三角 -INF）。
   这样 pad token 不参与任何位置的 softmax 分母。
3. **positions**：pad 位置的 position id 建议置 0（反正 mask 屏蔽），真实 token 从 0 递增。
4. **图形状**：`ids [max_len, batch]`、`positions [max_len, batch]`、
   输出 `hidden [n_embd, max_len, batch]` → 取 `hidden[:, -1, :]` → `[n_embd, batch]` → 逐条 MRL + L2 归一化。

#### 6.2 对外接口
```cpp
namespace Frostfall {

struct EmbedStats {
    double load_ms       = 0.0;   // 模型加载耗时
    double tokenize_ms   = 0.0;   // 分词耗时
    double compute_ms    = 0.0;   // graph 前向耗时
    double pool_ms       = 0.0;   // pooling + normalize 耗时
    int    n_tokens_max  = 0;     // 本次 batch 内最大 token 数
    int    n_texts       = 0;     // 本次 batch 大小
};

struct EmbedRequest {
    std::vector<std::string> texts;   // 一批文本
    bool         is_query   = false;  // true → 自动拼 Instruct:\nQuery: 前缀
    std::string  task;                // 空 → 用 EngineConfig.default_task
    int          target_dim = 0;      // 0 或 1024 → 满维；[32,1024) → MRL 截断
};

struct EmbedResponse {
    std::vector<std::vector<float>> embeddings;  // shape: [n_texts, target_dim]
    int          dim        = 0;
    EmbedStats   stats;
    std::string  error;               // 非空表示本次失败
};

class EmbeddingEngine {
public:
    struct EngineConfig {
        std::string model_path;
        int  n_ctx        = 8192;
        int  n_threads    = 8;
        int  max_batch    = 8;
        int  target_dim   = 1024;
        int  pad_token_id = 151643;
        std::string default_task =
            "Given a web search query, retrieve relevant passages that answer the query";
    };

    bool Init(const EngineConfig& cfg);
    bool InitFromJsonFile(const std::string& path);
    bool InitFromJsonString(const std::string& json);

    EmbedResponse Embed(const EmbedRequest& req);       // 主接口
    std::vector<float> Embed(const std::string& text,   // 便捷单条接口
                             bool is_query = false,
                             const std::string& task = "");

    EmbedStats GetLastStats() const;
    ~EmbeddingEngine();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace Frostfall
```

设计决策（对齐 `frostfall.llm` v1.0）：
- 采样类字段**不出现**（embedding 无采样概念）。
- 所有请求**无状态**：每次 `Embed` 独立完成，不留 KV、不留 hidden，可跨线程安全调用。
- `EmbeddingEngine` 内部一把 `std::mutex` 保护 GGUF/gallocr/临时 ctx；`Embed` 是**串行**执行
  （不做 continuous batching），并发调用通过 mutex 排队。
- **无 `Cancel`**：一次前向就结束，秒级返回，取消收益极小；如需，可 v1.1 加。
- **不引入 `stop` / `finish_reason`**：embedding 无生成，无这些概念。

#### 6.3 CMake 组织（单一共享库 + 模块目录，v0.1 已定型）

`src/` 按任务分三个模块目录，统一编进一个 `libfrostfall.so`；v1.0 只是往
`FROSTFALL_EMBED_SOURCES` 里追加 `embedding_engine.cpp`，不动 CMake 骨架：

```
src/core/    model.cpp tokenizer.cpp common.cpp              # llm / embed 共用
src/llm/     graph.cpp kv_cache.cpp sampler.cpp qwen3_chat.cpp inference_engine.cpp
src/embed/   embed_graph.cpp pooling.cpp qwen3_embed_prompt.cpp
             embedding_engine.cpp（v1.0 新增） embed_types.h（v1.0 新增）

examples/llm/api_test/       # llm 侧 4 个调用示例（现有）
examples/embed/
  v0_1_smoke/                # v0.1 遗留 CLI（仍保留可跑）
  embed_blocking.cpp         # v1.0 示例：单条 + query/doc + MRL
  embed_batch.cpp            # v1.0 示例：4 条示例 + 相似度矩阵与官方对拍
  embed_test_util.h          # v1.0 共享 utils（读参考向量文件、cosine 断言）
  embed_conf.json            # v1.0 默认配置
```

> **为什么不是两个库**：初版方案是 `libfrostfall`（llm）+ `libfrostfall_embed` 两静态库并存、
> 共用源文件各编一份。定稿改为单库：教学规模下"只部署 embed"的诉求并不存在，
> 单库让 examples 只需 `target_link_libraries(... frostfall)`；模块边界由 src 目录划分承担，
> 不靠库边界。将来若真要拆，保证 target 名 / include 目录 / 扁平包含方式三条契约不变即可平滑演进。

#### 6.4 测试用例（v1.0 收口验收）
覆盖矩阵（每个 test 内多组场景）：
1. 单条 query（带 instruction 模板）→ 与官方 cosine ≥ 0.999
2. 单条 document（无前缀）→ 与官方 cosine ≥ 0.999
3. batch=4（2 query + 2 document）→ 相似度矩阵与官方 diff ≤ 0.01
4. MRL：同一 batch，`target_dim=256/512/1024` 三档，各自 `‖v‖₂ ≈ 1`；
   高维 top-k 检索排名与官方一致
5. 长文本：接近 `n_ctx=8192` 的文本能正常返回，不 OOM 不越界
6. 无状态复用：同一 engine 实例连续 100 次调用，`GetLastStats` 每次独立、结果稳定
7. 多线程：4 线程并发调用 `Embed`，结果与串行调用**逐值一致**（mutex 串行化保障）

---

## 7. 目录结构（定稿）

复用现有 `frostfall.llm` 仓库，`src/` 按任务分模块目录，`examples/` 按产品线分目录（不新建仓库）：

```
frostfall.llm/                      # 仓库根不变
├── doc/
│   ├── llm/                        # 现有
│   ├── qwen_model/                 # 现有（模型架构资料）
│   └── embed/                      # 新增
│       ├── design_embed.md               # 本文档
│       ├── code_analysis_embed-v0.1.md   # v0.1 逐接口代码走读（对齐 llm 版风格）
│       └── convert_hf_to_gguf_qwen3-0.6b_embed.md  # embed 模型转换笔记
├── src/
│   ├── core/                       # 任务无关基础设施（llm / embed 共用）
│   │   ├── model.{h,cpp}           # GGUF 加载 + qwen3_model 权重（含 lm_head，embed 不用）
│   │   ├── tokenizer.{h,cpp}       # 自研 byte-level BPE
│   │   ├── common.{h,cpp}          # 计时 / 格式化工具
│   │   └── log.h
│   ├── llm/                        # 生成式推理专用（现有文件搬入，内容不动）
│   │   ├── inference_engine.{h,cpp} / llm_types.h
│   │   ├── qwen3_chat.{h,cpp}
│   │   ├── graph.{h,cpp}           # 带增量 KV cache 的前向图（embed 不用，独立保留）
│   │   ├── kv_cache.{h,cpp}
│   │   └── sampler.{h,cpp}
│   └── embed/                      # 文本向量专用
│       ├── embed_graph.{h,cpp}     # v0.1 新增（专用图，llm graph 独立保留）
│       ├── pooling.{h,cpp}         # v0.1 新增
│       ├── qwen3_embed_prompt.{h,cpp}  # v0.1 新增
│       ├── embed_types.h           # v1.0 新增
│       └── embedding_engine.{h,cpp}    # v1.0 新增
├── examples/
│   ├── llm/api_test/               # 现有：4 个 LLM 调用示例
│   └── embed/                      # 新增
│       ├── v0_1_smoke/             # v0.1 最小示例（单条 + instruction + MRL）
│       ├── embed_blocking.cpp      # v1.0 示例：单条 + query/doc + MRL
│       ├── embed_batch.cpp         # v1.0 示例：4 条示例 + 相似度矩阵与官方对拍
│       ├── embed_test_util.h       # v1.0 共享 utils
│       └── embed_conf.json         # v1.0 默认配置
├── scripts/
│   ├── (LLM 现有)
│   └── embed_reference.py          # 新增：调用 transformers 生成参考向量
├── models/
│   ├── qwen3-0.6b-f16.gguf                 # 现有
│   └── qwen3-embedding-0.6b-f16.gguf       # 新增（转换产物）
└── CMakeLists.txt                  # 单一共享库 libfrostfall.so（core + llm + embed 模块源文件）
                                    # option(FROSTFALL_BUILD_EXAMPLES ...) 统一控制示例构建
```

CMake 顶层：**一个共享库** `libfrostfall.so` 承载全部模块，examples 只链接 `frostfall` target。
模块边界靠目录划分与头文件包含关系维持，不靠库边界。将来若出现"只部署 embed"或
"只部署 llm"的诉求，再按模块边界拆库，examples 因三条契约（target 名 / include 目录 /
扁平包含方式）不变而无感。

---

## 8. 对拍验证方案

### 8.1 参考实现（Python）
用 `conda run -n llm` 环境，脚本 `scripts/embed_reference.py`：
- 走 `transformers` 路径（README §Transformers Usage），显式实现 `last_token_pool + F.normalize`。
- 输出文件格式：每行一条文本对应一行 JSON `{"text": "...", "dim": 1024, "v": [f0, f1, ...]}`。
- **不用 `sentence-transformers`** 作首选：因为它会隐式做 pooling + normalize + prompt 拼接，
  黑盒程度高；用裸 `transformers` 更贴近我们 C++ 侧的算子层实现，便于定位数值差异来源。
- v1.0 可再加一份 `sentence-transformers` 版对拍作**端到端**双保险。

### 8.2 对拍指标
- **首选：cosine 相似度**（对 F16 累积误差最宽容）。阈值 ≥ 0.999 视为通过。
- **辅助：逐维绝对误差**。前 8 维打印出来目视对齐；批量跑 ≤ 5e-3 报警。
- **端到端：相似度矩阵**。跑 README 官方 2×2 示例，与
  `[[0.7646, 0.1414], [0.1355, 0.6000]]` 元素级 diff ≤ 0.01（v0.1 逐条单跑，v1.0 一次 batch）。

### 8.3 常见坑（预警）
1. **右 padding → last-token pool 取到 pad**：v1.0 的 batch 路径务必用左 padding；若某种原因用右 padding，
   要改成"按 attention_mask 求真实序列长 - 1，索引到 hidden[b, seq_len[b]-1, :]"（官方 fallback）。
2. **忘了 L2 归一化**：会导致 cosine 计算需要额外除范数，与官方 dot product 值对不上。
3. **MRL 顺序错**：**先截断维度再归一化**，反了就不是单位向量。
4. **误加 chat 模板**：Embedding **不**过 ChatML；query 侧只加 `Instruct:\nQuery:` 前缀。
5. **`skip_special=false` vs `true`**：Embedding 阶段用 `encode` 就行，
   不涉及 `id_to_piece`，无该坑（LLM 版 v1.0 M3 的 `<think>` 吞噬坑此处不适用）。
6. **`output.weight` 张量读取错误**：转换脚本仍会写 `output.weight`（tie 但物理存），
   Embedding 版**跳过不读**，否则白占 300MB 内存且无用。
7. **causal 与双向混淆**：Embedding 版仍是 **causal**，不要把 mask 改成全 0。
8. **pad_token_id**：Qwen3-Embedding 官方无独立 pad，习惯用 `<|endoftext|>` (`151643`) 兼作 pad，
   从 GGUF `tokenizer.ggml.padding_token_id` 或 `tokenizer.ggml.eos_token_id` 读取，配置里可覆盖。

---

## 9. 里程碑速览

| 里程碑 | 交付物 | 关键验收 |
| --- | --- | --- |
| **M1 = v0.1** | `models/qwen3-embedding-0.6b-f16.gguf` + `src/embed/{embed_graph,pooling,qwen3_embed_prompt}` 新增 + `examples/embed/v0_1_smoke/` + `scripts/embed_reference.py` | 单条对拍 cosine ≥ 0.999；官方 2×2 相似度矩阵（逐条单跑）元素 diff ≤ 0.01；MRL 三档均为单位向量 |
| **M2 = v1.0** | `EmbeddingEngine` 收进 `libfrostfall.so`（无新库产物）+ `src/embed/{embed_types.h, embedding_engine.{h,cpp}}` + embed_graph 加 batch/attention mask + 两个示例 CLI + `embed_conf.json` | 4 条 batch 每条 cosine ≥ 0.999；官方 2×2 相似度矩阵（一次 batch）元素 diff ≤ 0.01；4 线程并发结果与串行逐值一致 |

---

## 10. 后续（不在本方案范围）
- **Reranker**（`Qwen3-Reranker-0.6B`）：走"yes/no logits 打分"的生成式路径，与本 Embedding 方案不同，
  预留 `frostfall.rerank` 子模块，不阻塞本方案。
- **量化**（Q4_K / Q8_0）：需要时新增 `qwen3-embedding-0.6b-q4_k.gguf`，
  推理端 `embed_graph.cpp` 无改动（ggml 自动分派）。
- **CUDA / Metal backend**：接口无需改动，只切换 ggml backend init。
- **持久化 batch server**：若要做在线服务，包一层薄 HTTP 即可（`EmbeddingEngine` 已线程安全）。
