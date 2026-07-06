# frostfall.llm —— 基于 ggml 的极简教学版大模型推理框架设计方案

> 目标：以 **Qwen3-0.6B** 为唯一目标模型，在 **ggml v0.15.3** 之上，用尽量少、尽量清晰的 C/C++ 代码，
> 实现一个"能跑通、看得懂、可扩展"的教学版推理框架，替代 llama.cpp 学习其核心原理。

---

## 1. 项目定位与设计原则

### 1.1 我们要做什么
- 只做 **推理（inference）**，不做训练。
- 只支持 **一个模型**：Qwen3-0.6B（GGUF 格式）。不做多架构抽象、不做通用化。
- 只跑 **CPU backend** 起步（ggml-cpu），第一版不碰 CUDA/Metal。
- 代码规模目标：主体链路控制在 **1000~1500 行 C++** 以内，便于逐行读懂。

### 1.2 我们刻意不做什么（相比 llama.cpp）
| llama.cpp 的复杂点 | 本项目的取舍 |
| --- | --- |
| 支持上百种模型架构 | 只写死 Qwen3 |
| 复杂的 KV cache 管理（分页、slot、defrag） | 一段连续内存的朴素 KV cache |
| 多种 backend 调度 + 张量并行 | 单 CPU backend |
| batched/parallel decoding、投机采样 | 单序列、单 batch |
| 完整 grammar / logit bias / 采样器链 | greedy + 基础 temperature/top-k/top-p |
| server / API / 量化工具链 | 只做命令行 demo |

### 1.3 设计原则
1. **可读性 > 性能 > 通用性**。宁可多写注释、少用宏魔法。
2. **分版本迭代**，每个版本都能独立编译运行、都有可验证的产出。
3. **紧贴 ggml 官方示例风格**，方便对照源码学习。
4. 每一步都能与 `llama.cpp` 对同一 prompt 的输出做**数值对拍**验证正确性。

---

## 2. Qwen3-0.6B 架构速览（实现前必须搞清楚）

Qwen3-0.6B 是一个标准的 decoder-only Transformer，但相比 GPT-2 有几个关键差异，
这些差异正是本项目要重点讲清楚的地方。

### 2.1 超参数（hparams）
| 参数 | 值 | 说明 |
| --- | --- | --- |
| `n_vocab` | 151936 | 词表大小 |
| `n_embd`（hidden_size） | 1024 | 隐藏维度 |
| `n_layer` | 28 | Transformer block 数 |
| `n_head` | 16 | Query 注意力头数 |
| `n_head_kv` | 8 | **KV 头数（GQA，分组查询注意力）** |
| `n_embd_head`（head_dim） | 128 | **每个头的维度（注意：≠ n_embd/n_head = 64）** |
| `n_ff`（intermediate_size） | 3072 | FFN 中间维度 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `rope_theta` | 1000000 | RoPE 频率基数 |
| `n_ctx_train` | 40960 | 训练上下文长度 |
| `tie_word_embeddings` | true | **lm_head 与 token embedding 权重共享** |

> ⚠️ 关键坑：`head_dim(128) * n_head(16) = 2048 ≠ n_embd(1024)`。
> 也就是说 Q/K/V 投影后的维度不是 hidden_size，需要严格按 head_dim 计算，不能想当然。

### 2.2 与 GPT-2 的核心区别（本项目的教学重点）
1. **归一化**：`RMSNorm`（无 bias、无减均值），而非 GPT-2 的 `LayerNorm`。
   - 用 `ggml_rms_norm` + 乘以 `weight`。
2. **位置编码**：`RoPE`（旋转位置编码，NEOX 变体），而非 GPT-2 的可学习位置嵌入 `wpe`。
   - 用 `ggml_rope_ext`，`mode = GGML_ROPE_TYPE_NEOX`。
3. **注意力**：`GQA`（16 个 Q 头共享 8 个 KV 头）。
   - KV cache 只存 8 个头，计算时通过 broadcast 让每 2 个 Q 头共享 1 组 KV。
4. **QK-Norm**（Qwen3 特色）：对每个头的 Q、K 在做 RoPE **之前** 各做一次 RMSNorm。
   - 权重 `attn_q_norm.weight`、`attn_k_norm.weight`，形状为 `[head_dim]`。
5. **FFN**：`SwiGLU`（`down(silu(gate(x)) * up(x))`），而非 GPT-2 的 `gelu`。
6. **无 bias**：Qwen3 的 Q/K/V/O 投影和 FFN 均无 bias（对照 GPT-2 大量 bias）。
7. **权重共享**：输出层 `lm_head` 直接复用 `token_embd.weight`。

### 2.3 单层（block）前向伪代码
```
# x: [n_embd, n_tokens]
h      = rms_norm(x) * attn_norm_w            # 输入归一化
q      = attn_q_w @ h                          # [head_dim*n_head,    n_tokens]
k      = attn_k_w @ h                          # [head_dim*n_head_kv, n_tokens]
v      = attn_v_w @ h                          # [head_dim*n_head_kv, n_tokens]

# reshape 成 [head_dim, n_head(_kv), n_tokens]
q = reshape(q, head_dim, n_head,    n_tokens)
k = reshape(k, head_dim, n_head_kv, n_tokens)
v = reshape(v, head_dim, n_head_kv, n_tokens)

# QK-Norm：对最后一维(head_dim)做 RMSNorm 后乘各自 norm 权重
q = rms_norm(q) * attn_q_norm_w
k = rms_norm(k) * attn_k_norm_w

# RoPE（NEOX），传入每个 token 的位置
q = rope(q, positions, theta=1e6)
k = rope(k, positions, theta=1e6)

# 写入 KV cache（把当前 k,v 追加到 cache 的 n_past 处）
cache_k[:, n_past:n_past+n_tokens] = k
cache_v[:, n_past:n_past+n_tokens] = v

# 注意力（GQA：K/V 头通过 broadcast 复用）
scores = (Q @ K^T) / sqrt(head_dim)            # [n_kv, n_tokens, n_head]
scores = soft_max_ext(scores, mask, scale)     # 因果 mask
attn   = scores @ V                            # [head_dim, n_tokens, n_head]
attn   = reshape(attn, n_embd_all, n_tokens)
x      = x + attn_output_w @ attn              # 残差

# FFN（SwiGLU）
h2 = rms_norm(x) * ffn_norm_w
x  = x + ffn_down_w @ ( silu(ffn_gate_w @ h2) * (ffn_up_w @ h2) )   # 残差
```
最后：`logits = lm_head @ (rms_norm(x_last) * output_norm_w)`。

---

## 3. ggml 基础知识速览（框架依赖的 4 个概念）

对照 `examples/gpt-2/main-backend.cpp` 与 `examples/simple/`：

1. **`ggml_context`**：张量与算子节点的内存池。加载权重用一个 ctx（no_alloc），
   每次前向构图用一个临时 ctx（只放节点元数据，不放数据）。
2. **`ggml_backend`**：计算后端。第一版用 `ggml_backend_cpu_init()`。
3. **`ggml_backend_buffer`**：真正存放张量数据的显存/内存块。权重、KV cache 各一块。
4. **计算图 `ggml_cgraph` + 分配器**：
   - 每步推理：新建 ctx → `build_graph` 搭出节点 → `ggml_gallocr_alloc_graph` 分配中间张量 →
     写入输入（token id、position、mask）→ `ggml_backend_graph_compute` 执行 → 读出 logits。
   - 分配器用 `ggml_gallocr`（对应 gpt-2 的 `main-alloc.cpp`/`main-backend.cpp`）。

关键算子清单（均已确认存在于 `include/ggml.h`）：
`ggml_get_rows`（embedding 查表）、`ggml_rms_norm`、`ggml_mul`、`ggml_mul_mat`、
`ggml_reshape_3d/4d`、`ggml_rope_ext`、`ggml_permute`、`ggml_cont`、
`ggml_soft_max_ext`（带 mask+scale）、`ggml_silu`、`ggml_add`、`ggml_cpy`/`ggml_view`（KV cache 写入）。

---

## 4. 整体模块划分

```
frostfall.llm/
├── doc/
│   └── design.md                # 本文档
├── src/
│   ├── main.cpp                 # CLI 入口：解析参数、跑生成循环
│   ├── gguf_loader.{h,cpp}      # 读 GGUF：hparams + 权重张量 + tokenizer 数据
│   ├── model.{h,cpp}            # qwen3_model 结构体、权重组织、backend/buffer 初始化
│   ├── graph.{h,cpp}            # build_graph：搭出 Qwen3 单步/多 token 前向计算图
│   ├── kv_cache.{h,cpp}         # 朴素连续 KV cache
│   ├── tokenizer.{h,cpp}        # BPE 分词/反分词（从 GGUF 内嵌词表构建）
│   ├── sampler.{h,cpp}          # greedy / temperature / top-k / top-p
│   └── common.{h,cpp}           # 参数解析、计时、日志等小工具
├── scripts/
│   ├── convert_hf_to_gguf.py    # HF → GGUF 转换脚本（拷贝自 llama.cpp）
│   ├── convert_hf_to_gguf_update.py
│   ├── gguf/                     # 转换脚本依赖的 gguf-py 包（拷贝自 llama.cpp）
│   └── encode_prompt.py         # v0.1 用：transformers 预处理 prompt → token id
├── models/                      # 放 qwen3-0.6b GGUF 文件（.gitignore）
├── third_party/ggml/            # ggml源码
├── CMakeLists.txt
└── README.md
```

> 说明：早期版本可以先把所有代码堆在 `main.cpp` 里跑通，等链路稳定后再按上表拆分文件——
> 这样避免一开始就陷入"接口设计"而忘了先让它跑起来。

---

## 5. 版本路线图（Roadmap）

按"先主干、后血肉"的顺序迭代，每一版都可编译、可运行、可验证。

| 版本 | 主题 | 产出 / 验收标准 |
| --- | --- | --- |
| **v0.1** | 跑通前向主体链路（单 token、贪心解码）+ 对拍验证 | 输入 prompt 能自回归生成通顺文本；与 llama.cpp 同 seed 贪心输出逐 token 一致 |
| **v0.2** | Tokenizer 自实现 + 增量 KV cache + 模块化/计时 | 不再依赖外部分词；prompt 一次性 prefill、decode 单 token 增量；代码拆模块、有 tokens/s 与内存统计 |
| **v0.3** | 采样策略 | 支持 temperature / top-k / top-p，可复现随机采样 |
| **v0.4** | 性能与后端扩展（选做，进阶） | 支持量化权重（Q4_K/Q8_0）、可选 CUDA backend、flash-attention |

> 工程化与可观测不单独成版，而是分散到各版本：**对拍脚本**是 v0.1 的验证工具（不能往后拖）；
> **模块拆分与计时统计**在 v0.2 自然发生（代码变胖 + 要证明提速）；README/注释/错误处理是各版本常规动作。

下面逐版本展开。

---

### v0.1 —— 主体链路（本项目的核心里程碑）

**目标**：把"加载模型 → 构图 → 前向 → 取 logits → argmax → 拼接 → 再前向"的闭环跑通，
并建立与 llama.cpp 的**对拍验证手段**（这是 v0.1 能否自证正确的前提，不能往后拖）。

为降低第一版难度，允许如下简化：
- **Tokenizer 先偷懒**：用 Python（transformers）把 prompt 提前编码成 token id 数组，
  写进一个文本文件，C++ 端直接读入；生成的 token id 也 dump 出来，用 Python decode 看结果。
  （分词器的完整实现推迟到 v0.2，避免第一版被 BPE 细节拖住。）
- **KV cache 先偷懒**：每步都把"已生成的全部 token"重新完整前向一遍（O(n²) 但最简单、最不易错）。
  真正的增量 KV cache 推迟到 v0.2。
- **只支持贪心解码**（argmax），不做采样。

**任务拆解**：
1. **GGUF 加载**（`gguf_init_from_file`）：
   - 读元数据填 `qwen3_hparams`（key 形如 `qwen3.embedding_length`、`qwen3.block_count`、
     `qwen3.attention.head_count`、`qwen3.attention.head_count_kv`、
     `qwen3.attention.key_length`(=head_dim)、`qwen3.attention.layer_norm_rms_epsilon`、
     `qwen3.rope.freq_base` 等）。
   - 建立 `name -> ggml_tensor*` 映射（张量名见 §7）。
   - 把权重数据从文件读入 backend buffer。
2. **模型结构体** `qwen3_model`：持有 hparams、各层权重指针、ctx、backend、buffer。
3. **构图** `build_graph(model, tokens, n_past)`：严格照 §2.3 伪代码用 ggml 算子搭出来。
   - 重点实现：RMSNorm、GQA 的 reshape/permute、QK-Norm、RoPE(NEOX)、
     因果 mask 的 `ggml_soft_max_ext`、SwiGLU、tied lm_head。
4. **推理循环**：读 token id → 构图 → 分配 → compute → 取最后一个位置的 logits → argmax →
   追加 token → 重复直到 EOS 或达到 `n_predict`。
5. **输出**：把生成的 token id 打印/写文件，用 Python 反分词验证是否通顺。
6. **对拍脚本**（v0.1 自带，非选做）：写 `scripts/` 下的对拍工具，同 prompt + `--temp 0`
   与 llama.cpp 逐 token 比对；不一致时能 dump 逐层张量定位（见 §8.4/§8.5）。

**验收**：
- 对同一段中/英文 prompt，贪心生成的 token id 序列与 `llama.cpp`（`--temp 0`）**逐 token 一致**。
- 若不一致，用"逐层张量对拍"定位（见 §8.5）。

---

### v0.2 —— 自研 Tokenizer + 增量 KV cache + 模块化

**目标**：去掉 v0.1 的两个"偷懒"，让框架自成闭环、且推理复杂度回到 O(n)；
同时借代码自然变胖的时机完成**模块拆分**与**基础计时/统计**。

1. **Tokenizer（BPE / byte-level BPE）**：
   - 从 GGUF 读 `tokenizer.ggml.tokens`（词表）、`tokenizer.ggml.merges`（合并规则）、
     `tokenizer.ggml.token_type`，以及特殊 token id（bos/eos/pad）。
   - 实现 Qwen 的 byte-level BPE：UTF-8 → byte → 贪心/优先级合并。
   - 实现 `encode(text) -> ids` 与 `decode(ids) -> text`。
   - 处理 Qwen3 的 chat 模板特殊 token（`<|im_start|>`/`<|im_end|>` 等，可选）。
2. **增量 KV cache**：
   - 预分配 `cache_k / cache_v`：形状 `[head_dim * n_head_kv, n_ctx]` × `n_layer`。
   - prefill 阶段：一次性把整段 prompt（n_tokens 个）前向，KV 写入 `[0, n_tokens)`。
   - decode 阶段：每步只前向 1 个新 token，K/V 追加写到 `n_past` 处（用 `ggml_view` + `ggml_cpy`）。
   - 注意力时 K/V 取 `[0, n_past+1)` 的有效区间。
   - 因果 mask 与 position 依据 `n_past` 正确设置。
3. **模块拆分**：把 v0.1 堆在 `main.cpp` 的代码按 §4 目录拆成
   `gguf_loader / model / graph / kv_cache / tokenizer / sampler / common`，明确头文件接口。
4. **计时/统计**：加载/prefill/decode 各阶段计时与 tokens/s；打印权重与 KV cache 内存占用
   （增量 KV cache 的卖点就是"比 v0.1 快"，必须能测量才能证明）。

**验收**：纯 C++ 端从文本 prompt 到文本输出全链路跑通；相同 prompt 下与 v0.1 输出一致但速度显著提升，且能打印 tokens/s。

---

### v0.3 —— 采样策略

- `sampler` 支持：`greedy`、`temperature`、`top-k`、`top-p`、可设随机种子。
- CLI 暴露 `--temp / --top-k / --top-p / --seed / --n-predict`。
- 重复惩罚（repeat penalty）作为可选项。

**验收**：固定 seed 下输出可复现；temp=0 时退化为贪心且与 v0.2 一致。

---

### v0.4 —— 性能与后端扩展（选做，进阶）

- **量化权重**：支持直接加载 Q8_0 / Q4_K GGUF（`ggml_mul_mat` 已原生支持量化 × f32）。
- **CUDA backend**：`#ifdef GGML_USE_CUDA` 切换 `ggml_backend_cuda_init()`（对照 gpt-2 示例）。
- **`ggml_backend_sched`** 多后端调度；**flash-attention**（`ggml_flash_attn_ext`）。
- batched / 多序列并行（对照 `main-batched.cpp`）。

---

## 6. 关键实现要点与易错点

### 6.1 GQA（分组查询注意力）的张量摆放
- Q reshape 到 `[head_dim, n_head, n_tokens]`，K/V reshape 到 `[head_dim, n_head_kv, n_tokens]`。
- `ggml_mul_mat` 支持 broadcast：当 K 的头维能整除 Q 的头维时（16 / 8 = 2），
  可让每 2 个 Q 头共享 1 组 KV，无需手动复制 KV。摆维度时把"头"放到第 3 维，参考 llama.cpp 的做法。

### 6.2 RoPE 的正确调用
- `ggml_rope_ext(ctx, x, pos, freq_factors=NULL, n_dims=head_dim, mode=GGML_ROPE_TYPE_NEOX, ...)`
- `pos` 是一个 `I32` 张量，长度 = n_tokens，值为每个 token 的绝对位置（decode 时 = n_past）。
- `freq_base` 传 1e6（来自 hparams），`freq_scale=1.0`。
- **必须用 NEOX 变体**（Qwen/LLaMA 系），用错 mode 会导致输出乱码但不报错——常见坑。

### 6.3 QK-Norm（Qwen3 专有）
- 在 reshape 成 `[head_dim, n_head, n_tokens]` 之后、RoPE 之前，
  对 Q、K 沿 head_dim 做 `ggml_rms_norm` 再乘 `attn_q_norm_w` / `attn_k_norm_w`。
- 顺序不能错：**先 QK-Norm 再 RoPE**。

### 6.4 因果 mask
- 用 `ggml_soft_max_ext(ctx, kq, mask, scale, max_bias=0)`：
  - `scale = 1/sqrt(head_dim)`。
  - `mask` 是 `[n_kv, n_tokens]` 的 F32 张量，上三角（未来位置）填 `-INF`，其余 0。
- decode 阶段 n_tokens=1 时 mask 全 0（当前 token 能看到所有历史）。

### 6.5 tied embedding（权重共享）
- GGUF 里 0.6B 通常没有独立 `output.weight`；若缺失则 lm_head 直接用 `token_embd.weight`。
- 加载时要判断：`output.weight` 存在则用之，否则 fallback 到 `token_embd.weight`。

### 6.6 数据类型
- 权重多为 F16 或量化类型；中间计算走 F32。`ggml_mul_mat` 自动处理混合精度。
- KV cache 第一版用 F16（省内存），需注意读写一致。

---

## 7. Qwen3 GGUF 张量命名对照（加载时用）

llama.cpp 转换脚本产出的标准命名（`blk.{i}` 为第 i 层）：

| 逻辑名 | GGUF 张量名 | 形状（约） |
| --- | --- | --- |
| token embedding | `token_embd.weight` | `[n_embd, n_vocab]` |
| 输入 RMSNorm | `blk.{i}.attn_norm.weight` | `[n_embd]` |
| Q 投影 | `blk.{i}.attn_q.weight` | `[n_embd, head_dim*n_head]` |
| K 投影 | `blk.{i}.attn_k.weight` | `[n_embd, head_dim*n_head_kv]` |
| V 投影 | `blk.{i}.attn_v.weight` | `[n_embd, head_dim*n_head_kv]` |
| **Q norm（QK-Norm）** | `blk.{i}.attn_q_norm.weight` | `[head_dim]` |
| **K norm（QK-Norm）** | `blk.{i}.attn_k_norm.weight` | `[head_dim]` |
| 输出投影 | `blk.{i}.attn_output.weight` | `[head_dim*n_head, n_embd]` |
| FFN RMSNorm | `blk.{i}.ffn_norm.weight` | `[n_embd]` |
| FFN gate | `blk.{i}.ffn_gate.weight` | `[n_embd, n_ff]` |
| FFN up | `blk.{i}.ffn_up.weight` | `[n_embd, n_ff]` |
| FFN down | `blk.{i}.ffn_down.weight` | `[n_ff, n_embd]` |
| 末端 RMSNorm | `output_norm.weight` | `[n_embd]` |
| lm_head | `output.weight`（可能缺失→复用 token_embd） | `[n_embd, n_vocab]` |

> 建议加载后立即打印所有张量名与形状，与上表核对（这是排错第一手段）。

---

## 8. 构建、运行与验证

### 8.1 依赖 ggml
- 以 **git submodule** 方式引入 ggml，版本锁定 **v0.15.3**：

```bash
git submodule add https://github.com/ggml-org/ggml.git third_party/ggml
cd third_party/ggml && git checkout v0.15.3 && cd ../..
git submodule update --init --recursive
```

- CMake 里 `add_subdirectory(third_party/ggml)`，链接 `ggml`、`ggml-cpu`
  （对照 `third_party/ggml/examples/gpt-2/CMakeLists.txt`）。
- 克隆本仓库时需带子模块：`git clone --recurse-submodules <repo>`。

### 8.2 准备模型（HF → GGUF 转换）
- 本地 HF 模型：`/home/data/model/Qwen3-0___6B/`（`torch_dtype=bfloat16`，单 `model.safetensors`）。
- ggml 自身**不提供** HF→GGUF 转换脚本，转换生态在 llama.cpp。
  用 `llama.cpp/convert_hf_to_gguf.py`（已确认支持 `Qwen3ForCausalLM`）：

```bash
python /home/data/llama.cpp/convert_hf_to_gguf.py \
  /home/data/model/Qwen3-0___6B/ \
  --outfile /home/data/frostfall.llm/models/qwen3-0.6b-f16.gguf \
  --outtype f16
```

- **精度：F16**。理由：ggml 对 F16 支持最成熟，`ggml_mul_mat` 原生处理「F16 权重 × F32 激活」，
  无需手写反量化；与 gpt-2 官方示例一致；0.6B 转 F16 约 1.2GB，内存无压力。量化(Q4_K/Q8_0)推迟到 v0.4。
- 转换会把 **tokenizer 词表/merges/特殊 token + chat 模板** 一并写入 GGUF 元数据，
  供 v0.1 的 chat 模板与 v0.2 的自研分词直接读取，无需额外文件。

### 8.3 运行
```
./frostfall -m models/qwen3-0.6b-f16.gguf -p "你好，介绍一下你自己" -n 128
```

### 8.4 正确性验证（强烈建议每版都做）
- **黑盒对拍**：同 prompt、`--temp 0`，与 `llama.cpp` 的 `llama-cli` 输出逐 token 对比。
- **白盒对拍（定位 bug 用）**：在两边分别 dump 第 0 层的 `h、q、k、attn、ffn` 张量，
  逐元素比较（允许 1e-2 量级误差，因 F16/累加顺序不同）。差异突然变大的那一步就是 bug 所在。

### 8.5 常见 bug 定位顺序
1. logits argmax 全错 → 检查张量名映射、tied embedding、维度 reshape。
2. 前几个 token 对、后面发散 → KV cache 写入位置 / position / mask 错误。
3. 输出乱码 → RoPE mode 用错（应为 NEOX）、或 QK-Norm 顺序错、或 scale 漏了。
4. 数值整体偏移 → RMSNorm eps 或 norm 权重乘错维度。

---

## 9. 学习路径建议

1. 先精读 `examples/gpt-2/main-backend.cpp`（约 300 行搭图逻辑），理解 ggml 的
   "ctx / backend / gallocr / cgraph" 四件套。
2. 再读 `examples/simple/ggml-simple-explained.md`，理解最小可运行单元。
3. 从 v0.1 开始，**先把 GPT-2 例子改造成能加载 GGUF**，再逐步把算子替换成 Qwen3 的
   （LayerNorm→RMSNorm、wpe→RoPE、gelu→SwiGLU、加 GQA、加 QK-Norm）。
   —— 这种"渐进式改造"比从零写更不容易错，也更能看清两种架构的差异。

---

## 10. Review 决定（已确认）

1. **Tokenizer**：v0.1 用 **Python(`transformers`) 预处理**把 prompt 编码成 token id 给 C++ 读；
   生成的 id 再用 Python `decode` 验证。自研 BPE 分词推迟到 v0.2（作为支线学习）。
2. **模型精度**：转 **F16 GGUF**（见 §8.2）。KV cache 第一版用 F16。
3. **Chat 模板**：**需要支持**。v0.1 起就套用 Qwen3 的 `<|im_start|>role\n...<|im_end|>` 模板
   （Python 预处理阶段用 `tokenizer.apply_chat_template` 生成即可；模板文本也在 GGUF 元数据中）。
4. **命名**：框架名与可执行文件名均为 **`frostfall`**。
5. **量化 / 多后端**：**暂不做**（v0.4 选做，当前不列入计划）。
