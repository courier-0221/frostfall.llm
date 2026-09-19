# asr/api_test —— asr/v0.1 数值对齐示例
#
# asr_align_v01：加载 qwen3-asr GGUF + tools/export_asr_reference.py 导出的参考
# 数据（work/asr_ref/{auto,lang}），验证 prefill 逐层 hidden / 末位 logits、
# teacher forcing decode、自由贪心生成的数值对齐。
# 参考数据不在构建目录内，运行时用 --ref-dir 指向工程根下的 work/asr_ref/<case>。

add_executable(asr_align_v01 asr_align_v01.cpp)
target_link_libraries(asr_align_v01 PRIVATE frostfall)
