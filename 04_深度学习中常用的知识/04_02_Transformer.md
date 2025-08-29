

# 1 Embedding 的定义

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

## 1.1 Embedding 的实现方式

### 1.1.1 🔹 方法 1：查表（Lookup Table）
![](image/Pasted%20image%2020250825122307.png)


### 1.1.2 方法 2：位置编码（Positional Encoding）

- Transformer 需要 **顺序信息**
- 在 token embedding 上加 **位置向量**，告诉模型 token 的顺序
- 常用方式：
    - 正弦/余弦编码 (sin/cos)
    - 可训练的位置向量 (learnable positional embedding)

## 1.2 Embedding 输出

- 假设：
    - 序列长度 = L token
    - embedding 维度 = d
- 输出矩阵：
    - ![](image/Pasted%20image%2020250825122708.png)
- 每一行就是对应 token 的向量表示，包含了**词语语义 + 位置信息**



## 1.3 例子 

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

# 2 **线性变换**过程: Q/K/V 矩阵（Query / Key / Value）

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






# 3 **LLM 中 Self-Attention 的流程示意图**，展示 token、embedding、Q/K/V 矩阵及 Attention 输出的关系：


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




# 4 优化器

https://blog.csdn.net/qq_53250079/article/details/128981653

在训练神经网络时，我们需要不断更新参数（权重）。  
更新公式由 **优化器 (optimizer)** 决定，比如：
- **SGD** (随机梯度下降)
- **Adam / AdamW**（深度学习里最常用）
优化器不是只存权重，它还会保存一些**额外的变量**来帮助训练更快收敛，这些变量就是 **优化器状态**。


## 4.1 Adam 优化器的状态

Adam 使用了 **动量 (momentum)** 和 **二阶动量 (variance)** 来平滑更新。

对于每一个模型参数 www：

- 权重本身：www
    
- 一阶矩（动量）：mmm
    
- 二阶矩（平方梯度的移动平均）：vvv
    

所以，除了存参数外，Adam 还要存 **m 和 v**。


## 4.2 显存占用对比

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
