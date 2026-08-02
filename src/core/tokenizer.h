#pragma once

// frostfall v0.2 —— 自研 Qwen3 分词器（byte-level BPE，等价于 HF 的 GPT2/Qwen2 tokenizer）。
//
// 去掉了 v0.1 对 Python(transformers) 的依赖：直接从 GGUF 元数据里读取
//   - tokenizer.ggml.tokens      词表（每个 token 的“字节级编码”字符串）
//   - tokenizer.ggml.merges      BPE 合并规则（按优先级排列）
//   - tokenizer.ggml.token_type  每个 token 的类型（NORMAL / CONTROL / ...）
//   - 特殊 token id（bos/eos/pad）
// 然后在 C++ 端实现 encode(text)->ids 与 decode(ids)->text。
//
// 关键概念（详见 tokenizer.cpp 顶部注释）：
//   1. byte-level：先把文本按 UTF-8 拆成字节，再用 GPT-2 的 bytes_to_unicode 把每个字节
//      映射成一个“可打印”的 Unicode 字符（例如空格 0x20 -> 'Ġ'）。词表里存的就是这些字符串。
//   2. pretokenize：用 Qwen2 的正则把文本先切成“词片段”，BPE 只在片段内部合并、不跨片段。
//   3. BPE：在每个片段里按 merges 的优先级不断合并相邻的最优 pair，直到无法再合并。

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct qwen3_tokenizer {
    // ---- 词表 ----
    std::vector<std::string> id_to_token; // 每个 id 的字符串（NORMAL：字节级编码；特殊 token：字面量如 "<|im_end|>"）
    std::vector<int32_t>     token_type;  // 与 id_to_token 对齐：1=NORMAL 3=CONTROL 4=USER_DEFINED ...
    std::unordered_map<std::string, int32_t> token_to_id;

    // BPE 合并规则：key = 左片段 + " " + 右片段（片段内不含空格字节，故用空格分隔无歧义），value = 优先级（越小越优先）
    std::unordered_map<std::string, int32_t> merge_rank;

    // 特殊 token（CONTROL / USER_DEFINED）的字面量 -> id，encode 时先按最长匹配把它们从文本中切出来
    std::vector<std::pair<std::string, int32_t>> special_tokens;

    int32_t eos_id = -1; // 生成停止符（Qwen3 = <|im_end|> 151645）
    int32_t bos_id = -1;
    int32_t pad_id = -1;
    bool    add_bos = false;

    // 从 GGUF 文件读取 tokenizer 元数据并构建上述结构。失败返回 false。
    bool load(const std::string & fname);

    // 文本 -> token id 序列。add_special=true 时会按 add_bos 决定是否前置 bos。
    // 输入里出现的特殊 token 字面量（如 "<|im_start|>"）会被识别并直接映射成对应 id。
    std::vector<int32_t> encode(const std::string & text) const;

    // token id 序列 -> 文本。skip_special=true 时跳过 CONTROL/USER_DEFINED token（不输出其字面量）。
    std::string decode(const std::vector<int32_t> & ids, bool skip_special = true) const;

    // 单个 token id -> 它对应的原始文本片段（供逐 token 流式打印用）。
    // 特殊 token：skip_special 时返回空串，否则返回字面量；NORMAL：把字节级编码还原成原始字节。
    std::string id_to_piece(int32_t id, bool skip_special = true) const;

    bool is_special(int32_t id) const {
        return id >= 0 && id < (int32_t) token_type.size() &&
               (token_type[id] == 3 /*CONTROL*/ || token_type[id] == 4 /*USER_DEFINED*/);
    }
};
