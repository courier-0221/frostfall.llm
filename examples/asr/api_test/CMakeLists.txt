# asr/api_test —— asr/v0.1 数值对齐示例
#
# asr_align_v01：加载 qwen3-asr GGUF + tools/export_asr_reference.py 导出的参考
# 数据（work/asr_ref/{auto,lang}），验证 prefill 逐层 hidden / 末位 logits、
# teacher forcing decode、自由贪心生成的数值对齐。
# 参考数据不在构建目录内，运行时用 --ref-dir 指向工程根下的 work/asr_ref/<case>。

# add_executable(asr_align_v01 asr_align_v01.cpp)
# target_link_libraries(asr_align_v01 PRIVATE frostfall)

# asr_align_v02：输入参考 Mel，C++ 完成音频 Encoder + ASR Prompt + 特征融合 +
# Decoder 生成，逐段对齐音频塔（conv_out / 18 层 / 投影）、融合 Embedding 与 Decoder 输出。
add_executable(asr_align_v02 asr_align_v02.cpp)
target_link_libraries(asr_align_v02 PRIVATE frostfall)
