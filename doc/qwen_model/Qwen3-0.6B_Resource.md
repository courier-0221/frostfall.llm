# Qwen3-0.6B 模型文件解析

## 目录结构

```text
Qwen3-0.6B
├── config.json
├── configuration.json
├── generation_config.json
├── merges.txt
├── model.safetensors
├── README.md
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

这是一个标准的 Hugging Face Transformers 格式模型目录。

从推理引擎的角度来看，可以分为三类资源：

```text
Qwen3-0.6B
├── 模型结构配置
├── Tokenizer资源
└── 模型权重
```

---

# 一、模型结构配置

## 1. config.json

最重要的配置文件，用于描述模型网络结构。

### 主要内容

```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "vocab_size": 151936,
  "max_position_embeddings": 32768
}
```

### 作用

描述：

* Hidden Size
* Transformer Layer数量
* Attention Head数量
* KV Head数量
* Vocabulary大小
* 最大上下文长度

对应模型结构：

```text
Embedding
    ↓
28层 Transformer Block
    ↓
LM Head
```

### 推理引擎中的作用

```cpp
LoadConfig("config.json");

Model model;
model.hidden_size = 1024;
model.layer_num = 28;
model.head_num = 16;
```

---

## 2. generation_config.json

模型默认生成参数配置。

### 示例

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "repetition_penalty": 1.05
}
```

### 作用

定义默认采样策略：

```cpp
session.temperature = 0.7;
session.top_p = 0.8;
session.top_k = 20;
```

### 注意

不影响模型结构，仅影响文本生成效果。

---

## 3. configuration.json

模型仓库附带的额外配置。

### 示例

```json
{
  "framework":"huggingface"
}
```

或者：

```json
{
  "model_type":"qwen3"
}
```

### 作用

通常用于模型管理工具识别模型。

实际推理过程中大多不会使用。

---

# 二、Tokenizer资源

Tokenizer负责：

```text
文本
 ↓
Token
 ↓
Token ID
```

以及反向过程：

```text
Token ID
 ↓
Token
 ↓
文本
```

---

## 4. vocab.json

词表文件。

### 示例

```json
{
  "你":1234,
  "好":5678,
  "Hello":345,
  "<|im_start|>":151643
}
```

### 作用

建立：

```text
Token ↔ ID
```

映射关系。

例如：

```text
你好
```

可能被编码为：

```text
[1234, 5678]
```

---

## 5. merges.txt

BPE(Byte Pair Encoding)合并规则。

### 示例

```text
h e
he l
hel lo
```

表示：

```text
h + e → he
he + l → hel
hel + lo → hello
```

### 作用

Tokenizer编码时：

```text
hello
```

不会拆成：

```text
h e l l o
```

而是：

```text
hello
```

作为一个Token。

### 优势

减少Token数量，提高推理效率。

---

## 6. tokenizer.json

完整Tokenizer定义文件。

包含：

```text
vocab
+
merges
+
normalizer
+
pre-tokenizer
+
special tokens
```

### 作用

现代Tokenizer通常直接加载：

```cpp
tokenizer.json
```

而不是：

```cpp
vocab.json
+
merges.txt
```

因为：

```text
tokenizer.json
=
vocab.json
+
merges.txt
+
其他规则
```

---

## 7. tokenizer_config.json

Tokenizer行为配置。

### 示例

```json
{
  "bos_token":"<|im_start|>",
  "eos_token":"<|im_end|>",
  "pad_token":"<|endoftext|>"
}
```

### 作用

定义：

* BOS（开始Token）
* EOS（结束Token）
* PAD（填充Token）

例如：

```cpp
tokenizer.eos_id();
```

就是从这里获取配置。

---

# 三、模型权重

## 8. model.safetensors

模型最核心的文件。

### 文件大小

```text
≈ 1.5GB
```

对应：

```text
Qwen3-0.6B
≈ 6亿参数
```

### 内容

保存所有神经网络权重：

```text
embed_tokens.weight

layers.0.attn.q_proj.weight
layers.0.attn.k_proj.weight
layers.0.attn.v_proj.weight

layers.1.xxx
...

lm_head.weight
```

### 推理时

```cpp
LoadWeights("model.safetensors");
```

加载到：

```text
Tensor
Weight Buffer
Memory Pool
```

中。

---

# 四、README.md

模型说明文档。

通常包含：

* 模型介绍
* Prompt格式
* 支持语言
* Benchmark结果
* License

### Qwen常见说明

```text
<|im_start|>system
...
<|im_end|>

<|im_start|>user
...
<|im_end|>
```

### 作用

端侧部署时重点查看：

* Prompt格式
* Chat模板
* 使用限制

---

# 五、推理时真正需要哪些文件

对于自研LLM推理引擎：

## 最小运行集合

```text
config.json
tokenizer.json
model.safetensors
```

即可完成推理。

---

# 六、完整推理流程

## 输入

```text
北京是中国的
```

---

## Step1 Tokenizer编码

加载：

```text
tokenizer.json
tokenizer_config.json
```

编码后：

```text
[8723, 1512, 374, ...]
```

---

## Step2 创建模型结构

读取：

```text
config.json
```

构建：

```text
28 Layers
1024 Hidden Size
16 Attention Heads
```

---

## Step3 加载权重

读取：

```text
model.safetensors
```

加载约：

```text
1.5GB权重
```

---

## Step4 Prefill阶段

```text
Prompt
 ↓
Embedding
 ↓
Transformer
 ↓
生成KV Cache
```

例如：

```text
北京
是
中国
的
```

全部进入模型计算。

---

## Step5 Decode阶段

利用KV Cache进行自回归解码：

```text
预测下一Token
```

例如：

```text
首
```

---

继续预测：

```text
都
```

---

## Step6 Detokenize

Token：

```text
首
都
```

转换为：

```text
首都
```

最终输出：

```text
北京是中国的首都
```

---

# 七、文件作用总结

| 文件                     | 作用            |
| ---------------------- | ------------- |
| config.json            | 定义模型结构        |
| configuration.json     | 模型仓库附加配置      |
| generation_config.json | 默认采样参数        |
| tokenizer.json         | 完整Tokenizer实现 |
| tokenizer_config.json  | Tokenizer行为配置 |
| vocab.json             | Token词表       |
| merges.txt             | BPE合并规则       |
| model.safetensors      | 模型权重          |
| README.md              | 模型说明文档        |

---

# 八、一句话理解

可以把整个模型理解成：

```text
config.json
    ↓
定义模型长什么样

tokenizer.json
    ↓
定义文字如何变成Token

model.safetensors
    ↓
定义模型学到了什么知识
```

推理过程：

```text
输入文本
    ↓
Tokenizer
    ↓
Token IDs
    ↓
LLM Forward
    ↓
Logits
    ↓
Sampling
    ↓
Token IDs
    ↓
Detokenize
    ↓
输出文本
```

对于端侧推理框架（如 llama.cpp、MLC-LLM、TensorRT-LLM、MTK NeuroPilot）来说，最核心的三个文件就是：

```text
config.json
tokenizer.json
model.safetensors
```

其余文件主要用于配置、兼容和辅助生成。
