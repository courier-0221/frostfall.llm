# api_test 端到端用例
#
# 每个 .cpp 编译成一个独立可执行文件，统一链接 frostfall 共享库。
# 由于本文件通过 add_subdirectory 引入，产物默认落在与源码镜像的构建目录：
#   build/tests/llm/api_test/
# 无需再手动设置 RUNTIME_OUTPUT_DIRECTORY。

set(FROSTFALL_API_TESTS
    infer_nothink_stream
    infer_nothink_blocking
    infer_yesthink_stream
    infer_yesthink_blocking
)

foreach(test_name IN LISTS FROSTFALL_API_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE frostfall)
endforeach()
