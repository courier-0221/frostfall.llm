#pragma once

// frostfall.embed v0.1 —— Qwen3-Embedding-0.6B 计算图。
//
// 与 llm 版 graph.h 的核心区别：
//   1. 无 KV cache：单次前向就结束，K/V 只作本次图的临时张量，随图释放。
//   2. 无 lm_head：图的最终输出是 final RMSNorm 后的 [n_embd, n_tokens] 隐层张量，
//      不再乘 model.output 得到 logits。
//   3. 无 n_past：位置直接从 0 开始；因果 mask 是标准的下三角。
//
// 保留：
//   - 28 层 block 的 RMSNorm / QKV 投影 / QK-Norm / RoPE-NEOX / GQA 因果注意力 / SwiGLU FFN
//     （逻辑与 llm 版 graph.cpp 完全一致，只是层内张量不写 KV cache）。
//   - causal mask：Qwen3-Embedding 基于 Qwen3-0.6B-Base（decoder-only causal），
//     last-token pool 的语义就是"最后位置能看见前面全部 token"，因此 mask 必须保留因果结构，
//     不能改成全 0 的双向 mask。

#include "model.h"

#include "ggml.h"

#include <cstdint>

// 图里两个输入张量、一个输出张量的固定名字：
// build 完图、gallocr 分配好之后，通过 ggml_graph_get_tensor() 按名字取出来填数据/读结果。
#define QWEN3_EMBED_TENSOR_NAME_TOKENS "tokens"    // I32 [n_tokens]              token id
#define QWEN3_EMBED_TENSOR_NAME_POS    "positions" // I32 [n_tokens]              RoPE 位置（0..n_tokens-1）
#define QWEN3_EMBED_TENSOR_NAME_MASK   "kq_mask"   // F32 [n_tokens, n_tokens]    因果 mask（未来位置填 -inf）
#define QWEN3_EMBED_TENSOR_NAME_HIDDEN "hidden"    // F32 [n_embd, n_tokens]      最后一层隐层输出（未经 lm_head）

// ctx 必须是 no_alloc=true 的临时 context（中间张量数据由调用方通过 ggml_gallocr 之后分配）。
// n_tokens：本次输入的 token 数。max_nodes：图节点上限（ggml_new_graph_custom）。
struct ggml_cgraph * qwen3_embed_build_graph(
        struct ggml_context * ctx,
        const qwen3_model    & model,
        int32_t                n_tokens,
        int                    max_nodes);
