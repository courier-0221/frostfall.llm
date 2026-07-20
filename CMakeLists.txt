cmake_minimum_required(VERSION 3.14)
project(frostfall LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json 供 VS Code IntelliSense 使用

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml 直接内置在 third_party/ggml 下（作为普通目录随本仓库跟踪，不使用 git submodule，见 doc/design.md §8.1）
add_subdirectory(third_party/ggml)

add_executable(frostfall
    src/main.cpp
    src/model.cpp
    src/graph.cpp
    src/kv_cache.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/common.cpp
)

target_include_directories(frostfall PRIVATE src)
target_link_libraries(frostfall PRIVATE ggml)
