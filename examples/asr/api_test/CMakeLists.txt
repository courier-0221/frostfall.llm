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
# add_executable(asr_align_v02 asr_align_v02.cpp)
# target_link_libraries(asr_align_v02 PRIVATE frostfall)

# asr_transcribe_v03：纯 C++ 离线转写示例（asr/v0.3）。输入 PCM16 WAV 或参考
# pcm.npy，经 src/asr_engine 的最小阻塞 API 得到语言与文字；--ref-dir 时对比
# Log-Mel 与生成 token 的 golden；--check-errors 演示显式报错路径。
# add_executable(asr_transcribe_v03 asr_transcribe_v03.cpp)
# target_link_libraries(asr_transcribe_v03 PRIVATE frostfall)
# configure_file(asr_conf_v03.json asr_conf_v03.json COPYONLY)

# asr_transcribe_v04：asr/v0.4 完整离线 API 示例。阻塞 vs 文本流式一致性、
# 回调取消 / 外部线程取消、--prefill-batch 分批 prefill、--window-ref 跨窗口
# golden 对齐（scripts/asr/export_asr_window_reference.py 导出）、--check-errors
# 错误分类（含上下文不足）与性能统计展示。
add_executable(asr_transcribe_v04 asr_transcribe_v04.cpp)
target_link_libraries(asr_transcribe_v04 PRIVATE frostfall)
configure_file(asr_conf_v04.json asr_conf_v04.json COPYONLY)
