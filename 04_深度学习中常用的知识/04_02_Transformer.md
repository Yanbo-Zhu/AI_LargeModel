

# 1 **Self-Attention**
## 1.1 普通 Attention 

## 1.2 PagedAttention（显存管理优化）


- LLM（如 LLaMA、GPT）使用 **Transformer 架构**，其中 **Self-Attention** 是核心计算：
    - 对输入序列长度为 LLL 的 token，注意力计算复杂度是 **O(L²)**
    - 长上下文（几万 token）时：
        - 计算量暴增
        - GPU 显存消耗极大
- 普通 Attention 会一次性把整个 **Q, K, V**（query/key/value）矩阵全部加载到显存 → 对大模型/长序列不可行


 PagedAttention 的核心思想
- **分页（Paging）机制）**：
    - 把长序列拆成 **多个小页（page）**
    - 每次只把一部分 Q/K/V 加载到显存
    - 分页计算 attention，然后逐页累加结果
- **优势**：
    1. **显存占用降低**
        - 不必一次性把整个序列加载到 GPU
        - 可以处理数万 token 的长上下文
    2. **支持大模型推理**
        - 即使模型和序列很大，也能在单张 GPU 上运行
    3. **按需调度 / 异步加载**
        - 可以边计算边加载下一页数据，提高吞吐率


想象一个 图书馆查找书籍：
普通 Attention = 把所有书都搬到桌子上，一次性查完 → 占空间多
PagedAttention = 一次只拿几本书查，查完放回，再拿下一批 → 空间少，但最终结果一样


vLLM 使用 PagedAttention 技术
- **显存优化** → 可以推理更长的上下文
- **批处理优化** → 同时处理多条请求
- **高吞吐量** → API 服务级别 LLM 推理

# 2 Embedding 的定义

- **Embedding** = 把离散的 token（文字/子词/单词）映射成 **连续的向量**
- 目的是让模型可以 **用数学运算处理文本信息**

简单理解
- Token 是符号或整数（离散）
- 神经网络只能处理 **连续数值向量**
- Embedding 就是 **把 token 转换成向量空间中的点**


- **文本 → Token**：把句子拆成最小单元
- **Token → Token ID**：每个 token 对应词表索引
- **Token ID → Embedding 向量**：查表得到向量
- **Embedding 矩阵**：每个 token 的向量按顺序排列，作为 Transformer 输入

## 2.1 Embedding 的实现方式

### 2.1.1 🔹 方法 1：查表（Lookup Table）
![](image/Pasted%20image%2020250825122307.png)


### 2.1.2 方法 2：位置编码（Positional Encoding）

- Transformer 需要 **顺序信息**
- 在 token embedding 上加 **位置向量**，告诉模型 token 的顺序
- 常用方式：
    - 正弦/余弦编码 (sin/cos)
    - 可训练的位置向量 (learnable positional embedding)

## 2.2 Embedding 输出

- 假设：
    - 序列长度 = L token
    - embedding 维度 = d
- 输出矩阵：
    - ![](image/Pasted%20image%2020250825122708.png)
- 每一行就是对应 token 的向量表示，包含了**词语语义 + 位置信息**



## 2.3 例子 

![](image/Pasted%20image%2020250825122800.png)


```
文本输入: "Hello world!"
           │
           ▼
Tokenization
["Hello", " world", "!"]
           │
           ▼
Token ID 映射
[17, 89, 5]
           │
           ▼
Embedding 查表 (Lookup Table)
E ∈ ℝ^(V×d)
           │
           ▼
得到向量表示
E_17 = [0.1, 0.5, 0.2, …]  (Hello)
E_89 = [0.3, 0.9, 0.4, …]  (world)
E_5  = [0.05, 0.12, 0.7, …] (!)
           │
           ▼
Embedding 矩阵 X ∈ ℝ^(L×d)
X = 
[ E_17 ]
[ E_89 ]
[ E_5  ]
(每行对应一个 token 的向量)

```

# 3 **线性变换**过程: Q/K/V 矩阵（Query / Key / Value）

- **位置**：在 Transformer 的 **Self-Attention 层**
- **作用**：用于计算 token 之间的注意力（Attention），决定每个 token 应该关注序列中的哪些部分


详细解释
1. **输入**：序列 embedding（每个 token 一个向量）
2. **线性变换**：
    - 将每个 token 的 embedding 映射成：
        - **Query (Q)**：表示“我想找什么信息”
        - **Key (K)**：表示“我拥有什么信息” 
        - **Value (V)**：表示“我实际提供的信息”
3. **计算注意力**： Attention(Q,K,V)=softmax(QKTdk)VAttention(Q, K, V) = softmax\Big(\frac{Q K^T}{\sqrt{d_k}}\Big) VAttention(Q,K,V)=softmax(dk​

![](image/Pasted%20image%2020250825122114.png)






# 4 **LLM 中 Self-Attention 的流程示意图**，展示 token、embedding、Q/K/V 矩阵及 Attention 输出的关系：


```
文本输入: "Hello world!"
           │
           ▼
Tokenization
["Hello", " world", "!"]
           │
           ▼
Embedding (每个 token → 向量)
E_1, E_2, E_3
           │
           ▼
线性变换 → 得到 Q/K/V
Q = [Q1, Q2, Q3]
K = [K1, K2, K3]
V = [V1, V2, V3]
           │
           ▼
Attention 计算
Attention(Q,K,V) = softmax(QK^T / √d_k) V
           │
           ▼
输出向量
O_1, O_2, O_3
(每个 token 的新表示，包含上下文信息)

```




# 5 优化器

https://blog.csdn.net/qq_53250079/article/details/128981653

在训练神经网络时，我们需要不断更新参数（权重）。  
更新公式由 **优化器 (optimizer)** 决定，比如：
- **SGD** (随机梯度下降)
- **Adam / AdamW**（深度学习里最常用）
优化器不是只存权重，它还会保存一些**额外的变量**来帮助训练更快收敛，这些变量就是 **优化器状态**。


## 5.1 Adam 优化器的状态

Adam 使用了 **动量 (momentum)** 和 **二阶动量 (variance)** 来平滑更新。

对于每一个模型参数 www：

- 权重本身：www
    
- 一阶矩（动量）：mmm
    
- 二阶矩（平方梯度的移动平均）：vvv
    

所以，除了存参数外，Adam 还要存 **m 和 v**。


## 5.2 显存占用对比

- **参数 (weights)**：大小 = 参数数 × dtype大小
    
- **梯度 (gradients)**：大小 ≈ 参数大小
    
- **Adam 状态 (m, v)**：≈ 2 × 参数大小

总显存≈参数+梯度+优化器状态≈1+1+2=4×参数大小

举例：
- LLaMA-70B 权重（FP16）：140GB
- 训练时需要 ≈ **140 × 4 = 560GB 显存**    

👉 这就是为什么训练大模型一定要用 **多机多卡 + ZeRO 优化 + checkpointing** 来分布存储。


- **优化器状态** = 优化器为每个参数额外保存的变量（例如 Adam 的 mmm 和 vvv）。
- 作用：帮助训练更稳定、更快收敛。
- 代价：显存占用通常是 **参数大小的 2–4 倍**。


# 6 tokenizer

在 **大模型（LLM, Large Language Model）** 中，**tokenizer（分词器）** 是非常关键的一步。它的作用就是把原始的文本（字符串）转换成模型可以理解和处理的 **token 序列**。

为什么需要 tokenizer？
- 神经网络不能直接处理文字（"我爱NLP"），只能处理数字（向量）。
- tokenizer 就是把自然语言转换成数字序列的“翻译器”。
- 如果没有 tokenizer，模型就不知道“一个词或字”对应什么数字。

用户输入文本 → tokenizer 编码 → token ID 序列
        → 大模型（Transformer 等）处理 → token ID 序列
        → tokenizer 解码 → 输出文本


## 6.1 Tokenizer 的主要作用

1. **切分文本 → token**
    - 把原始文本按规则切分成基本单元（token）。
    - token 可以是 **字符**、**词**、**子词（subword）**，甚至是字节。
    - 举例（英文 BPE 分词）：
        `"unbelievable" → ["un", "believ", "able"]`
    - 中文例子（SentencePiece）：
        `"我爱自然语言处理" → ["我", "爱", "自然", "语言", "处理"]`
2. **建立词表 (vocabulary)**
    - 给每个 token 分配一个唯一的 ID。
    - 例如：
        `"我" → 101 "爱" → 102 "自然" → 103`
        
3. **文本转数字序列**
    - 输入文本 → token ID 列表（整型序列）。
    - 例如：
        `"我爱自然语言处理" → [101, 102, 103, 104, 105]`
        
4. **处理特殊符号**
    - `"[CLS]"`：序列开始标记
    - `"[SEP]"`：序列分隔符
    - `"[PAD]"`：填充符（保证 batch 内长度一致）
    - `"[UNK]"`：未知词
5. **解码 (decode)**
    - tokenizer 还能把模型输出的 ID 序列翻译回自然语言。
    - 例如：
        `[101, 102, 103] → "我爱自然"`


## 6.2 不同 Tokenizer 的方法

1. **Word-level**（词级，早期方法，中文不适用）
    
2. **Character-level**（字符级，粒度太细，序列很长）
    
3. **Subword-level（主流）**
    
    - BPE（GPT-2, GPT-3, LLaMA 使用）
        
    - WordPiece（BERT 使用）
        
    - SentencePiece（T5, mT5, LLaMA2 使用）
        
    - Unigram LM
        
4. **Byte-level**
    
    - GPT-2 的 Byte-level BPE → 可以处理任何字符（表情符号、代码、混合语言）

