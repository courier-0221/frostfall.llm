add_executable(embed_blocking embed_blocking.cpp)
target_link_libraries(embed_blocking PRIVATE frostfall)

add_executable(embed_batch embed_batch.cpp)
target_link_libraries(embed_batch PRIVATE frostfall)
