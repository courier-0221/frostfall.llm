#pragma once

// 构建 Qwen3 单次前向的计算图（带增量 KV cache）。
//
// v0.1 相比 v1.0 的扩展（asr/v0.1，见 doc/asr/design_asr_v0.x.md §2.2 第 7 条）：
//   - 外部 Embedding 入口：params.embd 非空时替代 token 查表（get_rows），
//     直接以 [n_embd, n_tokens] 的 F32 融合 Embedding 作为第 0 层输入；
//   - 仅对最后位置执行 LM Head：prefill 不再产出 [n_vocab, n_tokens] 的巨大
//     logits，而是 view 最后一列后投影，logits 恒为 [n_vocab, 1]；
//     注意力/层计算不变，生成侧只读最后位置（与现有引擎行为一致）；
//   - 可选逐层输出导出：want_layer_outputs=true 时每层残差输出也作为图输出
//     （名字 layer_out_{i}），供 asr/v0.1 数值对齐工具逐层比对。

#include "kv_cache.h"
#include "model.h"

#include "ggml.h"

#include <cstdint>

// 图里输入/输出张量的固定名字：
// build 完图、gallocr 分配好之后，通过 ggml_graph_get_tensor() 按名字取出来填数据/读结果。
#define QWEN3_TENSOR_NAME_TOKENS "tokens"    // I32 [n_tokens]          token id（embd 入口时不设置数据）
#define QWEN3_TENSOR_NAME_EMBD   "embd"      // F32 [n_embd, n_tokens]  外部融合 Embedding（可选入口）
#define QWEN3_TENSOR_NAME_POS    "positions" // I32 [n_tokens]          RoPE 位置（值为 n_past + i）
#define QWEN3_TENSOR_NAME_MASK   "kq_mask"   // F32 [n_kv, n_tokens]    因果 mask（未来位置填 -inf），n_kv=n_past+n_tokens
#define QWEN3_TENSOR_NAME_LOGITS "logits"    // F32 [n_vocab, 1]        最后位置的输出 logits

// 逐层输出导出的张量名前缀（want_layer_outputs=true 时存在 layer_out_0 .. layer_out_{n_layer-1}）。
#define QWEN3_TENSOR_NAME_LAYER_OUT_PREFIX "layer_out_"

// 一次构图参数。embd 与 tokens 互斥：embd 非空时它必须指向调用方在 ctx 中创建
// （已 set_input）的 F32 [n_embd, n_tokens] 张量，直接作为第 0 层输入。
struct qwen3_graph_params {
    int32_t n_tokens = 0;
    int32_t n_past   = 0;
    int     max_nodes = 0;

    struct ggml_tensor * embd = nullptr; // 外部融合 Embedding [n_embd, n_tokens]（调用方创建），可空
    bool logits_last_only  = true;             // 仅对最后位置执行 LM Head（默认开）
    bool want_layer_outputs = false;           // 导出每层残差输出（对齐工具用）
};

// ctx 必须是 no_alloc=true 的临时 context（中间张量数据由调用方通过 ggml_gallocr 之后分配）。
// kv 的 cache 张量是持久的（在各自 buffer 里），本图会把新 K/V 写进去、并从中读取历史 K/V。
struct ggml_cgraph * qwen3_build_graph(
        struct ggml_context       * ctx,
        const qwen3_model         & model,
        const qwen3_kv_cache      & kv,
        const qwen3_graph_params  & params);
