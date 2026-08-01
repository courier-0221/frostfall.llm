# api_test 端到端调用示例
#
# 每个 .cpp 编译成一个独立可执行文件，统一链接 frostfall 共享库。
# 由于本文件通过 add_subdirectory 引入，产物默认落在与源码镜像的构建目录：
#   build/examples/llm/api_test/
# 无需再手动设置 RUNTIME_OUTPUT_DIRECTORY。

set(FROSTFALL_API_EXAMPLES
    infer_nothink_stream
    infer_nothink_blocking
    infer_yesthink_stream
    infer_yesthink_blocking
)

foreach(example_name IN LISTS FROSTFALL_API_EXAMPLES)
    add_executable(${example_name} ${example_name}.cpp)
    target_link_libraries(${example_name} PRIVATE frostfall)
endforeach()

# 把示例配置拷贝到可执行文件所在目录（build/examples/llm/api_test/），
# 方便直接以 --config llm_infer_conf.json 运行；json 有改动时 cmake 会自动重配并更新拷贝。
configure_file(llm_infer_conf.json llm_infer_conf.json COPYONLY)
