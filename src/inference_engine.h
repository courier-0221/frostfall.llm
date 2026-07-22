#pragma once

#include "llm_types.h"

#include <memory>
#include <string>

namespace Frostfall {

/// 推理性能统计（库内部统计，供调试 / 测试使用，不属于 LlmResponse 协议）
struct InferenceStats {
    int    prompt_token_count    = 0;  ///< prompt 经 tokenizer 后的 token 数
    int    gen_token_count       = 0;  ///< 本次自回归生成的 token 数（不含首 token，含 stop token）
    double prompt_tokens_per_sec = 0;  ///< prefill 吞吐 (tok/s)，分子按 promptTokenBatchSize 向上对齐
    double gen_tokens_per_sec    = 0;  ///< decode 吞吐 (tok/s)，gen_token_count / decode 累计耗时
    double ttft_ms               = 0;  ///< Time To First Token (ms)：infer 入口到产出首 token
};

///   {
///     "model_path": "models/qwen3-0.6b-f16.gguf",
///     "n_ctx": 4096, "n_threads": 4, "max_tokens": 512,
///     "sampling": { "temp": 0.7, "top_k": 20, "top_p": 0.8, "repeat_penalty": 1.1, "seed": 42 }
///   }
struct EngineConfig {
    std::string    model_path;              ///< GGUF 路径（必填）
    int32_t        n_ctx      = 4096;        ///< KV cache 上限（会被 n_ctx_train 截断）
    int32_t        n_threads  = 4;           ///< CPU 线程数
    int32_t        max_tokens = 512;         ///< 请求未指定 max_tokens 时的默认上限
    SamplingParams sampling;                 ///< 引擎级固定采样参数（不做请求级覆盖）
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /// 初始化（加载模型 + tokenizer，并采用传入的推理配置）
    ///
    /// 三选一入口，内部最终都归一到 EngineConfig：
    /// @param config      直接传入结构体配置
    bool Init(const EngineConfig& config);
    /// @param path        JSON 配置文件路径，内部读取并解析成 EngineConfig
    bool InitFromJsonFile(const std::string& path);
    /// @param json        JSON 配置字符串，内部直接解析成 EngineConfig
    bool InitFromJsonString(const std::string& json);

    /// 统一推理接口（流式与非流式合并）
    ///
    /// 调用语义：接口对上层表现为无状态请求响应。
    ///   业务每次传入完整的 messages（含多轮历史），
    ///   库内部在每次 infer 入口自动重置 KV Cache，不跨调用保留会话状态。
    ///
    /// 线程模型：
    ///   - `Init` / `Infer` 互斥，串行执行；
    ///     在一个线程调用 `Infer()` 时，另一线程调用上述方法会阻塞直到本次推理结束。
    ///   - `on_chunk` 在当前调用 `Infer()` 的线程里同步触发；
    ///     回调内禁止再调用同一 InferenceEngine 的 Init/Infer（会死锁）。
    ///     回调内允许调用 `cancel()`、拷贝 response、写日志等无锁操作。
    ///   - 推理过程中可由任意线程调用 `cancel()` 提前结束本次生成，
    ///     此时返回的 LlmResponse.finish_reason == CANCELLED，
    ///     已生成的 thinking / content / tool_calls 仍会正常累积与回调。
    ///
    /// @param request   推理请求；stream 字段决定行为模式
    /// @param on_chunk  流式回调；request.stream=false 时忽略，可传 {}
    /// @return          最终 LlmResponse；is_end 恒为 true，
    ///                  message.content / thinking 为本次推理累积后的完整文本，
    ///                  msg_id 与 request.msg_id 一致
    LlmResponse Infer(const LlmRequest& request,
                      const StreamCallback& on_chunk = {});

    /// 请求取消当前正在进行的 Infer()。
    ///
    /// 线程安全：可在任意线程（包括 StreamCallback 内部）调用，本方法不持锁、立即返回。
    /// 语义：仅设置一个原子标志；正在执行的 Infer() 会在下一次生成 step 检查到后退出，
    ///       返回的 LlmResponse.finish_reason 为 CANCELLED。
    ///       若调用时没有正在执行的 Infer()，标志会在下一次 Infer() 入口被重置，无副作用。
    void Cancel();

    /// 获取最近一次推理的性能统计（非接口协议，仅供调试）
    ///
    /// 线程安全：内部加锁读取最近一次 Infer() 回填的统计。
    InferenceStats GetLastStats() const;

    /// 当前生效的生成上限（EngineConfig.max_tokens）
    int MaxTokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace Frostfall
