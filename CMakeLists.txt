cmake_minimum_required(VERSION 3.14)
project(frostfall LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json 供 VS Code IntelliSense 使用

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml 直接内置在 third_party/ggml 下（暂不使用 git submodule 管理，见 doc/design.md §8.1）
add_subdirectory(third_party/ggml)

# glog v0.7.1（git submodule），关闭可选依赖以简化构建
set(WITH_GFLAGS   OFF CACHE BOOL "" FORCE)
set(WITH_GTEST    OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glog)

add_executable(frostfall
    src/main.cpp
    src/model.cpp
    src/graph.cpp
)

target_include_directories(frostfall PRIVATE src)
target_link_libraries(frostfall PRIVATE ggml glog::glog)
