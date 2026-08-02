#include "qwen3_embed_prompt.h"

const char * const QWEN3_EMBED_DEFAULT_QUERY_TASK =
    "Given a web search query, retrieve relevant passages that answer the query";

std::string qwen3_embed_build_query_text(const std::string & text,
                                         const std::string & task) {
    const std::string & t = task.empty() ? std::string(QWEN3_EMBED_DEFAULT_QUERY_TASK) : task;
    // 注意：官方模板 "Instruct: {task}\nQuery:{text}"，Query 冒号后没有空格。
    return "Instruct: " + t + "\nQuery:" + text;
}
