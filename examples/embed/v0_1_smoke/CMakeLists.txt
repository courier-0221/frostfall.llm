# frostfall.embed v0.1 —— 最小 smoke CLI
#
# embed 专属模块（embed_graph / pooling / qwen3_embed_prompt）已收进 frostfall 共享库，
# 本目录只剩一个薄壳 CLI。产物路径: build/examples/embed/v0_1_smoke/

add_executable(v0_1_smoke v0_1_smoke.cpp)
target_link_libraries(v0_1_smoke PRIVATE frostfall)
