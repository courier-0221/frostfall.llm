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

set(FROSTFALL_CORE_SOURCES
    src/model.cpp
    src/graph.cpp
    src/kv_cache.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/common.cpp
    src/qwen3_chat.cpp
    src/inference_engine.cpp
)

add_library(frostfall SHARED ${FROSTFALL_CORE_SOURCES})
# 对外头文件在 src/；nlohmann/json 在 third_party/。PUBLIC 让链接者自动获得头路径与依赖。
target_include_directories(frostfall PUBLIC src third_party)
target_link_libraries(frostfall PUBLIC ggml Threads::Threads)

# ---- 测试：链接 frostfall 库的使用示例 / 用例 ----
option(FROSTFALL_BUILD_TESTS "Build frostfall test executables" ON)
if (FROSTFALL_BUILD_TESTS)
    add_subdirectory(tests/llm/api_test)
endif()
