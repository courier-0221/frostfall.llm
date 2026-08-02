#pragma once

// frostfall.embed v0.1 —— Qwen3-Embedding query instruction 模板。
//
// 官方规则（读自 /home/lil72/data/models/Qwen3-Embedding-0.6B/config_sentence_transformers.json）：
//   - query 侧文本前拼：  "Instruct: " + task + "\nQuery:" + text
//   - document 侧文本：   原文，不加任何前缀
//   - 默认 task = "Given a web search query, retrieve relevant passages that answer the query"
//
// 与 llm 的 ChatML 完全不同：没有 <|im_start|> / <|im_end|> 包裹，没有 role，
// 前缀是纯文本，参与 BPE 分词，与用户 query 拼在同一段里。

#include <string>

// 内置的默认 task 描述（来自 config_sentence_transformers.json：prompts.query）。
extern const char * const QWEN3_EMBED_DEFAULT_QUERY_TASK;

// 构造 query 侧文本：拼上 "Instruct: {task}\nQuery:{text}"。
// task 为空时用 QWEN3_EMBED_DEFAULT_QUERY_TASK。
std::string qwen3_embed_build_query_text(const std::string & text,
                                         const std::string & task);
