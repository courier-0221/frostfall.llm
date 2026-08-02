cmake_minimum_required(VERSION 3.14)
project(frostfall LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json 供 VS Code IntelliSense 使用

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_subdirectory(third_party/ggml)

# ---- 模块源文件：src 按任务分层 ----
#   core/  任务无关基础设施（GGUF 加载 / 分词 / 工具），llm 与 embed 共用
#   llm/   生成式推理专用（增量 KV cache / 采样 / ChatML / 引擎门面）
#   embed/ 文本向量专用（单次前向图 / pooling / query 前缀模板）
set(FROSTFALL_CORE_SOURCES
    src/core/model.cpp
    src/core/tokenizer.cpp
    src/core/common.cpp
)
set(FROSTFALL_LLM_SOURCES
    src/llm/graph.cpp
    src/llm/kv_cache.cpp
    src/llm/sampler.cpp
    src/llm/qwen3_chat.cpp
    src/llm/inference_engine.cpp
)
set(FROSTFALL_EMBED_SOURCES
    src/embed/embed_graph.cpp
    src/embed/pooling.cpp
    src/embed/qwen3_embed_prompt.cpp
)

# 单一共享库：目录划分只是物理组织，产物仍是一个 libfrostfall.so。
add_library(frostfall SHARED
    ${FROSTFALL_CORE_SOURCES}
    ${FROSTFALL_LLM_SOURCES}
    ${FROSTFALL_EMBED_SOURCES}
)
# 对外头文件按模块目录 PUBLIC 暴露；nlohmann/json 在 third_party/。
# 调用方保持扁平包含（#include "model.h"），src 内部目录调整不影响 examples。
target_include_directories(frostfall PUBLIC src/core src/llm src/embed third_party)
target_link_libraries(frostfall PUBLIC ggml Threads::Threads)

# ---- 调用示例：链接 frostfall 库的端到端示例程序 ----
option(FROSTFALL_BUILD_EXAMPLES "Build frostfall example executables" ON)
if (FROSTFALL_BUILD_EXAMPLES)
    add_subdirectory(examples/llm/api_test)
    add_subdirectory(examples/embed/v0_1_smoke)
endif()
