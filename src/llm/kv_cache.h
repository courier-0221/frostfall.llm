#pragma once

// frostfall v0.2 —— 朴素的增量 KV cache。
//
// v0.1 每步都把“已生成的全部 token”重新完整前向一遍（O(n^2)）。v0.2 改为：
//   - prefill：一次性前向整段 prompt，把每层的 K/V 写入 cache 的 [0, n_prompt)。
//   - decode ：每步只前向 1 个新 token，把它的 K/V 追加写到 cache 的 n_past 处；
//              注意力时读取 cache 的 [0, n_past+1)。复杂度回到 O(n)。
//
// 存储布局（每层各一块，类型 F16 省内存）：
//   cache_k[il] / cache_v[il] 都是 1D 张量，长度 = n_embd_kv_all * n_ctx，
//   其中 n_embd_kv_all = head_dim * n_head_kv（GQA 只存 KV 头）。
//   逻辑上按 token 顺序摆放：第 t 个 token 占用 [t*n_embd_kv_all, (t+1)*n_embd_kv_all)。
//   读取时用 ggml_view_3d 把它看成 [head_dim, n_kv, n_head_kv]（见 graph.cpp）。

#include "model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

struct qwen3_kv_cache {
    int32_t n_ctx  = 0; // 预分配的上下文长度上限
    int32_t n_past = 0; // 当前 cache 中已有效的 token 数（= 下一次写入的起始位置）

    std::vector<struct ggml_tensor *> k; // 每层一个，[n_embd_kv_all * n_ctx]，F16
    std::vector<struct ggml_tensor *> v;

    struct ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buffer = nullptr;

    qwen3_kv_cache() = default;
    qwen3_kv_cache(const qwen3_kv_cache &) = delete;
    qwen3_kv_cache & operator=(const qwen3_kv_cache &) = delete;
    ~qwen3_kv_cache();

    // 预分配 n_layer * 2 块 cache 到 model.backend。n_ctx 为上下文上限。失败返回 false。
    bool init(const qwen3_model & model, int32_t n_ctx);

    // 已分配的 cache 内存字节数（用于统计打印）。
    size_t size_bytes() const;
};
