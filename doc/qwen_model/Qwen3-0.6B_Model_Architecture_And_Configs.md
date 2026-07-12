# Qwen3-0.6B 模型架构与 config.json 参数详解

## 一、Qwen3-0.6B整体结构

Qwen3-0.6B 属于 **Decoder Only Transformer**。

整体流程：

``` text
                 输入token
                     │
             Tokenizer编码
                     │
             token id (int)
                     │
──────────────────────────────────────────────
              输入层（Embedding）
──────────────────────────────────────────────

      vocab_size=151936
                │
        Embedding Matrix
      (151936 × 1024)
                │
      hidden_size =1024
                │
        加RoPE位置编码
                │
──────────────────────────────────────────────
        Transformer Block × 28
──────────────────────────────────────────────

 Block0
 ┌──────────────────────────┐
 │ RMSNorm                  │
 │ Multi Head Attention      │
 │ Residual                  │
 │ RMSNorm                  │
 │ MLP(SwiGLU)              │
 │ Residual                  │
 └──────────────────────────┘

 Block1

 ...

 Block27

──────────────────────────────────────────────
              输出层
──────────────────────────────────────────────

      RMSNorm
          │
     LM Head
          │
 logits(151936)
          │
     Softmax
          │
    预测下一个token
```

------------------------------------------------------------------------

## 二、config.json 参数对应关系

  可以把它整理成下面这个表。
| 参数                    | 所属模块                   |
| ----------------------- | -------------------------- |
| vocab_size              | 输入Embedding、输出LM Head |
| hidden_size             | 整个模型宽度               |
| num_hidden_layers       | Transformer层数            |
| num_attention_heads     | Attention                  |
| num_key_value_heads     | GQA Attention              |
| head_dim                | 每个Head维度               |
| intermediate_size       | MLP                        |
| hidden_act              | MLP激活函数                |
| rms_norm_eps            | RMSNorm                    |
| rope_theta              | RoPE                       |
| max_position_embeddings | 最大上下文                 |
| use_cache               | KV Cache                   |
| tie_word_embeddings     | 输入输出Embedding共享      |
| attention_dropout       | Attention Dropout          |
| attention_bias          | Linear是否Bias             |
| bos/eos_token           | 特殊token                  |
| torch_dtype             | 推理数据类型               |

------------------------------------------------------------------------

## 三、输入层

### vocab_size

``` json
"vocab_size":151936
```

Embedding矩阵大小：

``` text
151936 × 1024
```

每个Token都会映射为1024维向量。

### hidden_size

``` json
"hidden_size":1024
```

表示整个Transformer中所有隐藏向量宽度都是1024。

例如：

输入：

```
Hello
```

Embedding以后

```
(SeqLen,1024)
```

Attention输出：

```
(SeqLen,1024)
```

MLP输出：

```
(SeqLen,1024)
```

Residual：

```
(SeqLen,1024)
```

整个网络一直保持1024。

所以hidden_size就是模型宽度。

### max_position_embeddings

``` json
40960
```

表示模型最大支持40960个Token上下文。

### rope_theta

``` json
1000000
```

RoPE位置编码参数。

相比Llama的10000，更适合长上下文。

### rope_scaling

``` json
null
```

表示没有开启RoPE扩展。

------------------------------------------------------------------------

## 四、Transformer Block

``` json
"num_hidden_layers":28
```

表示共有28层Transformer Block。

每层结构：

``` text
Input
 │
RMSNorm
 │
Attention
 │
Residual
 │
RMSNorm
 │
MLP
 │
Residual
 │
Output
```

------------------------------------------------------------------------

### Attention

#### num_attention_heads

```
16
```

表示：

Attention分成16个Head。

因为：

```
hidden_size=1024
```

所以：

```
1024
÷
16
=
64
```

正常应该每个Head 64维。

但是Qwen这里又指定了：

```
head_dim=128
```

为什么还能成立？

因为Qwen采用的是 **GQA（Grouped Query Attention）**，查询头（Q）和键值头（K/V）的组织方式不同，不再简单满足 `hidden_size ÷ num_attention_heads = head_dim` 这一经典关系。实际实现中，Q、K、V 会分别通过线性层映射到各自需要的维度，再按 head_dim 切分。

------

#### head_dim

```
128
```

表示：

每个Attention Head内部计算时：

```
Q

128维

K

128维

V

128维
```

Attention就是：

```
QK^T

↓

Softmax

↓

V
```

都是128维。

------

#### num_key_value_heads

```
8
```

这是Qwen采用：

> GQA（Grouped Query Attention）

传统Transformer：

```
16个Q

16个K

16个V
```

Qwen：

```
16个Query

8个Key

8个Value
```

即：

```
QQQQQQQQQQQQQQQQ

KKKKKKKK

VVVVVVVV
```

多个Q共享一个KV。

优点：

KV Cache减半。

所以：

推理速度更快。

------

#### attention_bias

```
false
```

表示：

Attention里面Linear：

```
W*x
```

没有Bias：

```
W*x+b
```

中的：

```
b
```

被去掉了。

可以减少参数。

------

#### attention_dropout

```
0
```

训练时：

Attention不会Drop。

推理时本来也不用Drop。

------------------------------------------------------------------------

### MLP

Attention后就是MLP。

结构：

```
1024

↓

3072

↓

SwiGLU

↓

1024
```

------

#### intermediate_size

```
3072
```

就是：

MLP隐藏层大小。

第一层Linear：

```
1024

↓

3072
```

第二层：

```
3072

↓

1024
```

------

#### hidden_act

```
silu
```

激活函数：

SiLU。

Qwen实际上使用的是 **SwiGLU** 结构，SiLU 是其中门控分支使用的激活函数，因此配置里写的是 `"silu"`。

------------------------------------------------------------------------

### Normalization

#### rms_norm_eps

```
1e-6
```

RMSNorm：

```
x

↓

Normalize

↓

输出
```

为了避免：

```
除0
```

加：

```
1e-6
```

这个epsilon。

------------------------------------------------------------------------

## 五、输出层

Transformer最后：

```
1024维
```

经过：

LM Head。

------

### tie_word_embeddings

```
true
```

表示：

输入Embedding：

```
Embedding Matrix
```

输出LM Head：

```
Projection Matrix
```

共享同一套权重。

即：

```
Input

↓

Embedding

↓

Transformer

↓

Embedding^T

↓

Logits
```

这样：

参数更少。

------------------------------------------------------------------------

## 六、KV Cache

### use_cache

``` json
true
```

推理阶段缓存Key和Value，避免重复计算。

------------------------------------------------------------------------

## 七、特殊Token

```
bos_token_id
```

开始token。

例如：

```
<|im_start|>
```

对应某个ID（具体特殊 token 的文本形式取决于聊天模板）。

------

```
eos_token_id
```

结束token。

模型预测到它：

```
停止生成
```

------------------------------------------------------------------------

## 八、其它参数

### torch_dtype

``` json
"bfloat16"
```

模型权重采用BF16。

### initializer_range

``` json
0.02
```

训练初始化标准差。

### architectures

``` text
Qwen3ForCausalLM
```

说明模型属于Decoder Only Causal Language Model。

### model_type

``` text
qwen3
```

Transformers据此加载对应实现。

### max_window_layers

与Sliding Window Attention有关。

由于：

``` json
"use_sliding_window":false
```

因此当前模型不会使用。

------------------------------------------------------------------------

# 九、完整结构图

``` text
                 token ids
                     │
        vocab_size =151936
                     │
        Embedding Matrix
      (151936 × 1024)
                     │
      hidden_size=1024
                     │
      + RoPE(theta=1000000)
                     │
──────────────────────────────────────────
        Transformer ×28
(num_hidden_layers=28)
──────────────────────────────────────────

RMSNorm
(rms_norm_eps)

↓

Attention
 num_attention_heads=16
 num_key_value_heads=8
 head_dim=128
 attention_bias=false
 attention_dropout=0

↓

Residual

↓

RMSNorm

↓

MLP
1024
 ↓
3072
(intermediate_size)
 ↓
SiLU/SwiGLU
(hidden_act)
 ↓
1024

↓

Residual

────────────×28────────────

↓

Final RMSNorm

↓

LM Head
(tie_word_embeddings=true)

↓

Logits
(vocab_size=151936)

↓

Softmax

↓

Next Token
```

## 学习建议

考虑到你目前正在基于 **ggml 实现一个教学版的 Qwen3 推理框架**，建议按下面的顺序理解模型，每一步都能与 `config.json` 中的参数对应起来：

1. **Embedding 层**：理解 `vocab_size`、`hidden_size`。
2. **RoPE 位置编码**：理解 `max_position_embeddings`、`rope_theta`。
3. **Attention**：重点掌握 `num_attention_heads`、`num_key_value_heads`、`head_dim`，以及 Qwen 的 GQA 实现。
4. **MLP（SwiGLU）**：理解 `intermediate_size` 和 `hidden_act`。
5. **RMSNorm 与残差连接**：理解 `rms_norm_eps` 的作用。
6. **LM Head 与采样**：理解 `tie_word_embeddings`、`use_cache` 以及输出 logits 到生成 token 的流程。

对于实现教学版ggml推理框架，这些模块就是整个推理引擎的核心。
