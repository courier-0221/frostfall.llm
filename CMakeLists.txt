cmake_minimum_required(VERSION 3.14)
project(frostfall LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml 直接内置在 third_party/ggml 下（暂不使用 git submodule 管理，见 doc/design.md §8.1）
add_subdirectory(third_party/ggml)

add_executable(frostfall
    src/main.cpp
    src/model.cpp
    src/graph.cpp
)

target_include_directories(frostfall PRIVATE src)
target_link_libraries(frostfall PRIVATE ggml)
