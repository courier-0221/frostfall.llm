#pragma once

// 构建 Qwen3 单步前向的计算图。
//
// v0.1 的简化：没有增量 KV cache，每次调用都用从位置 0 到 n_tokens-1 的“完整序列”重新构图、
// 重新计算一遍（O(n^2)，但最简单、最不容易写错）。真正的增量 KV cache 推迟到 v0.2。

#include "model.h"

#include "ggml.h"

#include <cstdint>

// 图里三个输入张量、一个输出张量的固定名字：
// build 完图、gallocr 分配好之后，通过 ggml_graph_get_tensor() 按名字取出来填数据/读结果。
#define QWEN3_TENSOR_NAME_TOKENS "tokens"    // I32 [n_tokens]            token id
#define QWEN3_TENSOR_NAME_POS    "positions" // I32 [n_tokens]            RoPE 位置（本版本恒为 0..n_tokens-1）
#define QWEN3_TENSOR_NAME_MASK   "kq_mask"   // F32 [n_tokens, n_tokens]  因果 mask（未来位置填 -inf）
#define QWEN3_TENSOR_NAME_LOGITS "logits"    // F32 [n_vocab, n_tokens]   输出 logits

// ctx 必须是 no_alloc=true 的临时 context（张量的实际数据由调用方通过 ggml_gallocr 之后分配）。
// max_nodes 用于创建计算图（ggml_new_graph_custom），需要 >= 图中实际的算子节点数。
struct ggml_cgraph * qwen3_build_graph(
        struct ggml_context * ctx,
        const qwen3_model    & model,
        int32_t                n_tokens,
        int                    max_nodes);
