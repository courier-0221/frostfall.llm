### 最终目标

学完后，你应该能够：

- 看懂 Qwen3 的 config.json
- 手写 Embedding → Attention → MLP → LM Head
- 理解 KV Cache 的数据结构
- 用 ggml 实现一个最小可运行的 Qwen 推理 Demo
- 看懂 llama.cpp 中 Qwen 的核心代码

### 第1章：Qwen3-0.6B 整体架构

![大模型 | QWen3 结构解析 - 地平线智能驾驶开发者 - 博客园](https://images.openai.com/static-rsc-4/OGZHHxsbczdT_4h1uLEzt0OSM_njBEc9Hj2Wf5Z73yHrHFE9sJZfdfdfRdLgwGzWF23NS5ingp1oK8v_MnpjrbFH2CRzKC_8uz6J27YQlBc_ehO03WPYnHPX87U-qxiYoHJyelRc_8Va9wxNuy1TIf_-ZFB1TkfH9TN9dqeEGG9MjZDp4gQ46xN3wY-jUny7?purpose=fullsize)

### 学习内容

- 什么是 Decoder Only Transformer
- Qwen3 的整体数据流
- 为什么 LLM 是“预测下一个 Token”
- 0.6B 参数量是怎么来的

### 第2章：输入层（Tokenizer + Embedding）

![Understanding Multimodal LLMs | Sebastian Raschka, PhD](https://images.openai.com/static-rsc-4/ixOQn1rZ0ctTMSlx4mMwbnEAQEyqKfV5cxUB9S1ygG7I83y0fz4MtnsMdWkhXw8yXK16_6_59qjj2ToeAmGwEJsfX2_ap7zTDXpp2Y8TYrNFLMUwHJad16Yq5OLmF1krFiNYeTBfk6oYiudXDeRkcydBbeRzQ8LjXEv1gUv6RhTM3VL7dMAJMaM7Hu6RzvfJ?purpose=fullsize)

### 核心参数

| 参数         | 作用       |
| ------------ | ---------- |
| vocab_size   | 词表大小   |
| hidden_size  | 向量维度   |
| bos_token_id | 开始 Token |
| eos_token_id | 结束 Token |

### 第3章：RoPE 位置编码

![Decoding Llama3: Part 4 - Rotary Positional Embeddings – Decoding Llama3: An explainer for tinkerers](https://images.openai.com/static-rsc-4/9QGCGNRtc6PSKUh1Gbw4RLeuPsDYm4LrPwP0f94UeJ2Pm13cLOspc6lTdW06uTVIotpaiJM1GaZ5qvCnJ65OfWtAPHZJ6tAnyaLVZeYo24C-dYkadgOlJiz0Md01MAQ8P-GjiAPAYADVUGg3DTTkqbAR0Vz0aoLE6T1noW6Of0ZBGT2YE38EPujwUf2myiIq?purpose=fullsize)

### 重点理解

- 为什么 Transformer 不知道顺序
- RoPE 如何旋转向量
- rope_theta = 1000000 的意义
- 长上下文为什么更稳定

### 第4章：Attention（最核心）

![探秘Transformer系列之（27）--- MQA & GQA - 罗西的思考 - 博客园](https://images.openai.com/static-rsc-4/CDFa4p6FuLfXTqt6h5xz1ohPIB3H0ppP_Gst_fHoYWNnTguOqLraSpi7KgtAKVvO_S9TiXjsHWMz5pTetr82NzTH0cbZqJE21Go7nAwm8AnoqbWTxkmHx_2MvcWnqqxY7Z2O5ewYdMhKYxcesZPWC-Y1fvsL3s0iwpg2pB6Z8bd1Ubzgl1tnvQjIj0QFqwov?purpose=fullsize)

### 重点参数

| 参数                | 含义               |
| ------------------- | ------------------ |
| num_attention_heads | Query Head 数      |
| num_key_value_heads | KV Head 数         |
| head_dim            | 每个 Head 的维度   |
| attention_bias      | Linear 是否带 Bias |



### 这一章必须真正吃透

你需要能手算：

- Q 的 shape
- K 的 shape
- V 的 shape
- Attention Score 的 shape
- GQA 为什么能减少 KV Cache

### 第5章：KV Cache

![What Is KV Cache in LLMs? A 2026 Guide.](https://images.openai.com/static-rsc-4/cQmLcH_Hs3omMCRQSJqe2WoDoT5bVAfYj0xyJ1pzndz-_6mfTspQh0JuQItiXv6MQV75syxlb0ezAbEa0EBBAspzNP0IpAVFeM0cEFuPBqIYG4H3ujZhts9re6VYEikBdSez-tQ04gZdO4cdBZ1gaGmCKQFpYaiwIuMaWmUH9egI79kZAyD-8eS1yqLeG4M6?purpose=fullsize)

### 这是端侧推理最关键的一章

### 你要理解

- Prefill 阶段
- Decode 阶段
- 为什么首 Token 慢
- 为什么后续 Token 快
- KV Cache 的内存布局
- Qwen3-0.6B 大约需要多少 KV 内存

### 第6章：MLP（SwiGLU）

![8 Types of LLM Architectures Patterns You Should Understand | by agus abdul rahman | Medium | Medium](https://images.openai.com/static-rsc-4/ntLDWeoi6XjlWhcAtjIUHNROeSfGkCGxDdjqxNDPnvEGmANMTHw2BRssNM8JEyBm-lqY5zz1Ydxk10WqtBEnicmQBgW-X8LTjXFas2YpffvDKhxgh7gEKotPcPCOTE4c-eGAnCunpgjHAZE7ItZPmcd2cjj6HH22OTjCfId6KUnSXS7T4JIfThqGA_mPwyqy?purpose=fullsize)

### 核心参数

| 参数              | 作用         |
| ----------------- | ------------ |
| intermediate_size | MLP 扩展维度 |
| hidden_act        | SiLU 激活    |

### 第7章：RMSNorm

![LLM Fundamentals - Visualizing Transformer Internals (Chinese Simplified)](https://images.openai.com/static-rsc-4/Vgvq_SOn0iDV4AmGF3u5IcjNlSMdu5oFHbwLdTzx-LXPQ_oQ95J8kEfBscbASyK_QtZce5dcsErCeSROf6pK9PXgLtkoi-XelW53t7XwfAMuF-7wlQGaa6O8S3NL05bh-8vkZpIW5bngeU3iP5nqND_8BwGv3K9QjP1_1r1yJNQLmhyuHYna-1BCjjRhbNBy?purpose=fullsize)

### 重点

- 为什么不用 LayerNorm
- rms_norm_eps 的作用
- Qwen 与 Llama 的归一化设计

### 第8章：输出层与采样

![What is Temperature in LLMs? - by Avi Chawla](https://images.openai.com/static-rsc-4/XdC0u0ALkc80rrerd3YsLlxCRM4EWz4O7Xyo1nMfvLCKZuPCXTG3m-Z5RHt4up-GYpl4tTpbgTew70eKe9Ju5pP3PiBO6AF-rCS_vGQ-b5iyo8UsQ_Bh2pVy_yW6fnvnPlE3eZ9K2bvNB3WVIOHhyQzAhtjd0vYFiZaPKs6KifNPH2zanRTDhTk1Rdk2uPzU?purpose=fullsize)

### 你要理解

- LM Head
- tie_word_embeddings
- Logits
- Softmax
- Temperature
- Top-k
- Top-p
- Repetition Penalty

### 第9章：config.json 全参数逐项解析



### 这部分就是你最开始问的内容

每个参数都会回答：

- 它属于哪一层？
- shape 是什么？
- 在推理时什么时候被读取？
- 在 ggml 中对应哪个张量？

### 第10章：自己实现最小 Qwen 推理框架



### 最终工程结构（推荐）

mini_qwen/

├── tokenizer.cpp

├── model_loader.cpp

├── rope.cpp

├── attention.cpp

├── mlp.cpp

├── rmsnorm.cpp

├── kvcache.cpp

├── sampler.cpp

└── main.cpp

### 第11章：对照 llama.cpp 源码

![llama.cpp 源码解析_llama cpp-CSDN博客](https://images.openai.com/static-rsc-4/Fb-cZTwp-yThP1euNbu3oW9EdciJSA8wRIryRHWj2u4iIa97HpmYWPqSalCndJu3eRmSIp9yvGMHMj8NDXsDCV7sxXREExVjRuOuDxwDLb_h0UPPeG3ZSz3ZGx9Fo86bI9GujqmpV1seZun5Q7Zy5CMpHO1fie9sVO5Igk59JBJOekpO6Dxz9F_jUgSSQn7s?purpose=fullsize)

### 重点看

- llama_model_loader
- llama_build_graph
- llama_decode_internal
- llama_kv_cache
- Qwen 的 GQA 实现



### 推荐学习顺序（非常重要）

第1周：Embedding → RoPE → RMSNorm

第2周：Attention → GQA → KV Cache

第3周：MLP → LM Head → 采样 → 自己写 Demo

最后：阅读 llama.cpp 的 Qwen 实现

### 给你一个小测试

### 如果你能回答下面 5 个问题，就说明已经真正理解了

- 为什么 hidden_size=1024？
- 为什么 16 个 Query Head + 8 个 KV Head？
- KV Cache 为什么能加速？
- RoPE 为什么比绝对位置编码更适合长文本？
- LM Head 为什么可以和 Embedding 共享权重？