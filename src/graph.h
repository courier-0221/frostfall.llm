#pragma once

// 构建 Qwen3 单次前向的计算图（v0.2：带增量 KV cache）。
//
// 与 v0.1 的区别：
//   - build_graph 只对“本次输入的 n_tokens 个 token”构图（prefill 时 = 整段 prompt，decode 时 = 1）。
//   - 每层把新算出的 K/V 写入 kv cache 的 [n_past, n_past+n_tokens)，注意力读取 cache 的 [0, n_kv)。
//   - RoPE 位置、因果 mask 都以 n_past 为基准（见 main.cpp 里填充输入的部分）。

#include "kv_cache.h"
#include "model.h"

#include "ggml.h"

#include <cstdint>

// 图里三个输入张量、一个输出张量的固定名字：
// build 完图、gallocr 分配好之后，通过 ggml_graph_get_tensor() 按名字取出来填数据/读结果。
#define QWEN3_TENSOR_NAME_TOKENS "tokens"    // I32 [n_tokens]          token id
#define QWEN3_TENSOR_NAME_POS    "positions" // I32 [n_tokens]          RoPE 位置（值为 n_past + i）
#define QWEN3_TENSOR_NAME_MASK   "kq_mask"   // F32 [n_kv, n_tokens]    因果 mask（未来位置填 -inf），n_kv=n_past+n_tokens
#define QWEN3_TENSOR_NAME_LOGITS "logits"    // F32 [n_vocab, n_tokens] 输出 logits

// ctx 必须是 no_alloc=true 的临时 context（中间张量数据由调用方通过 ggml_gallocr 之后分配）。
// kv 的 cache 张量是持久的（在各自 buffer 里），本图会把新 K/V 写进去、并从中读取历史 K/V。
// n_past：cache 中已有的 token 数（本次输入之前）。max_nodes：图节点上限（ggml_new_graph_custom）。
struct ggml_cgraph * qwen3_build_graph(
        struct ggml_context * ctx,
        const qwen3_model    & model,
        const qwen3_kv_cache & kv,
        int32_t                n_tokens,
        int32_t                n_past,
        int                    max_nodes);
