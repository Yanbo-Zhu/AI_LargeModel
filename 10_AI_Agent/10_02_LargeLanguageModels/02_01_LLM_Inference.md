
# 1 INTRO 

## 1.1 WHAT MAKES LLMS SUITABLE FOR AI AGENTS?

Language as the interface
▪ Instructions and programming via natural language ▪ Results in natural language

Book me a holiday
Booking to Fiji in winter is confirmed


---

Reasoning capability*
▪ Encode pattern of reasoning, problem-solving and decision-making ▪ Handle open-ended instructions
If I book a holiday, then I first
      book a flight and then the hotel


---

Context awareness
▪ Consider context across a long dialogues
▪ Tracking goals and adaption to evolving context

I know you like Fiji from past dialogues,
                     so I try to book a Fiji holiday
Quantas is booked out, I'll try Fiji Airways instead


----

Few-shot & zero-shot learning
▪ Generalize from instructions 
▪ Learn from few examples


I never booked a holiday but I try to figure out how Example 1: Books a seaside hotel in Nadi Example 2: Find a family-friendly apartment on Waya
Learned: [feature] + [accommodation type] + [location]

---


Tool use & planning
▪ Output structured data (e.g., JSON)
▪ Decide which external tools to use when for which task ▪ Combines reasoning with action

{"start":"Berlin"; "goal":"Nadi"}
 Call Quantas API with endpoint www.quantas.com/api
  If first query the hotel API in Fiji, then I call
            the booking.com API and compare prices


## 1.2 


Natural Language Processing (NLP)
▪ Technology to handle human language using machines
▪ Help with human-to-machine communication (e.g., question/answer, AI agents) 
▪ Help with human-to-human communication (e.g., translation, AI agents)
▪ Help with machine-to-machine communication (e.g., AI agents)
▪ Understand, interpret and generate language


What makes it so difficult?
▪ Different languages, different linguistic structures
▪ Wrongly spelled words, wrong grammar, missing words ▪ Conjugations, negations, metaphor, analogies, etc.
▪ Language is highly context-sensitive

---


![](image/Pasted%20image%2020260220075718.png)




# 2 FORMER MODELS 

n-gram → Log-linear → FNN → RNN → LSTM → Transformer

Language modeling is the art of determining the probability of a sequence of words.  The goal of a language model is to determine the probability of a word sequence


Bag of Words
▪ For text classification
▪ Lookup the one-hot vector per word
▪ Count number of words appearances by making the sum of one-hot vectors
▪ Multiply by weight vector to get the score for a specific property
▪ If score exceeds a threshold, the text exhibits the property under investigation  若得分超过阈值，则文本具有所考察的属性


Limitations
▪ Not context sensitive
▪ Rare words receive less consideration

![](image/Pasted%20image%2020260220081022.png)

----

## 2.1 TYPES OF LANGUAGE MODELS
![](image/Pasted%20image%2020260220081336.png)


1 Masked (Bidirectional) LM
Predict intentionally hidden words within a text sequence by using the surrounding left and right context.
通过利用左右两侧的上下文信息，预测文本序列中被有意隐藏（mask）的词语。
说明：
模型会看到句子的左右内容
预测中间被遮挡的词
属于"填空式"训练方式
典型代表：BERT


2 Autoregressive LM
Generates text sequentially by predicting each word based on all previously generated words.
通过基于所有之前已经生成的词，逐词顺序地预测下一个词，从而生成文本。
从左到右生成
每一步依赖前面所有生成内容
是目前大语言模型（如 GPT）的主流架构


3 Non-autoregressive LM
Generate all words in parallel, assuming conditional independence between them given the input.
在给定输入的条件下，假设各个词之间条件独立，并行地一次性生成所有词。
不按顺序生成
所有词同时预测
速度更快，但通常质量不如自回归模型



4 Other
Examples:
▪ Non-context-aware LMs
不考虑上下文信息的语言模型。
说明
每个词的表示是固定的
不会根据上下文变化
典型例子：Word2Vec、GloVe


## 2.2 UNIGRAM LANGUAGE MODELS 一元语言模型



Independence Assumption
▪ The next word does not depend upon the current or former words
▪ Count-based maximum-likelihood estimation (MLE)
▪ Predicts next word based on MLE, but samples from probability distribution


Limitations
▪ No word order considered
▪ No context considered
▪ Unknown words (not in training set) not generated

独立性假设
下一个词不依赖于当前词或之前的词
基于计数的最大似然估计
根据最大似然估计预测下一个词，但从概率分布中进行采样

局限性
不考虑词序
不考虑上下文
无法生成未知词（即未出现在训练集中的词）


![](image/Pasted%20image%2020260220081651.png)


---


通过数数，然后用除法来算出概率，最后选择能使观察数据出现可能性最大的那个概率值。

![](image/Pasted%20image%2020260220081836.png)


## 2.3 AUTO-REGRESSIVE LANGUAGE MODELS

![](image/Pasted%20image%2020260220082336.png)


## 2.4 HIGHER-ORDER N-GRAM LANGUAGE MODELS

No Independence Assumption
▪ Limit the context length 𝑛 (bigram: 2, trigram: 3,...)
▪ Apply count-based maximum-likelihood estimation (MLE)
▪ Divide counts 

![](image/Pasted%20image%2020260220082452.png)



Limitations
▪ Still problem with unknown words (not in training set) not generated
▪ Each word treated independently even if similar meaning
▪ Can't deal with intervening words
▪ No long-distance dependencies
![](image/Pasted%20image%2020260220082524.png)

## 2.5 FEATURIZED LANGUAGE MODELS

Log-linear Model
▪ Model learns which features of the context are predictive
▪ Context feature vector depends on position relative to next word
▪ Look up context feature vectors and add a bias
▪ Calculate probabilities based on the scores
▪ Optimize feature weights using gradient descent
▪ 模型学习哪些上下文特征对预测是有用的
▪ 上下文特征向量依赖于相对于下一个词的位置
▪ 查找（lookup）上下文特征向量，并加入一个偏置项（bias）
▪ 基于计算得到的分数（scores）来计算概率
▪ 使用梯度下降（gradient descent）来优化特征权重


Limitations
Still no long-distance dependencies
Different words with similar meaning are not captured
▪ 仍然无法建模长距离依赖关系
▪ 无法捕捉语义相似的不同词语（相似含义的词不会被建模为接近）

![](image/Pasted%20image%2020260220082720.png)



## 2.6 LANGUAGE MODELS WITH ARTIFICIAL NEURAL NETWORKS


1 Feed-Forward Neural Network (FNN)
▪ Lookup word/token embeddings and concatenate
▪ Let FNN derive scores
▪ Apply softmax to derive probabilities
▪ Weight matrix captures combinations of features (not possible with log-linear models)
▪ Optimize feature weights using gradient descent
▪ 查找（lookup）词 / token 的嵌入向量（embeddings），并将它们拼接（concatenate）起来
▪ 让前馈神经网络（FNN）计算得分（scores）
▪ 使用 softmax 将得分转换为概率
▪ 权重矩阵能够捕捉特征的组合关系（这是对数线性模型无法做到的）
▪ 使用梯度下降（gradient descent）来优化特征权重


Limitations
▪ Still no long-distance dependencies 
▪ Fixed sequence length
▪ 仍然无法建模长距离依赖关系
▪ 序列长度是固定的（Fixed sequence length）


![](image/Pasted%20image%2020260220082857.png)


---

2 Recurrent Models
Recurrent Neural Networks (RNNs) condition representations on the history of words processed

Limitations: Short term memory, no parallelization for learning, Vanishing/exploding gradients (still challenge with very long dependencies)

![](image/Pasted%20image%2020260220083214.png)



----

3  Convolution Models

Convolutional Neural Networks (CNN) condition representations on the local context. Architecture like feed-forward network.

![](image/Pasted%20image%2020260220083328.png)



---


4 Long Short-term Memory (LSTM)
LSTM is a special kind of RNN with a cell state. The core idea is to memorize with gate keeping. Selectively remembering or forgetting inputs as needed. Three gates: forget, input and output gate
Forget gate: Decides if information is discarded from the cell state
Input gate: Controls which new values are added to the cell state
Output gate: Regulates how much of the cell state is exposed as the output
Limitation: No parallel processing because output depends on previous step. This 
results in longer training times. Poorly scales with length of sequence.

LSTM 是一种特殊类型的 RNN（循环神经网络），它包含一个 细胞状态（cell state）。
其核心思想是通过"门控机制（gate keeping）"进行记忆管理，根据需要选择性地记住或遗忘输入信息。

LSTM 包含三个门：

🔹 遗忘门（Forget Gate）
决定是否从细胞状态中丢弃某些信息。

🔹 输入门（Input Gate）
控制哪些新的信息被加入到细胞状态中。

🔹 输出门（Output Gate）
调节细胞状态中有多少信息被作为当前输出。


局限性（Limitations）
▪ 无法进行并行计算，因为当前输出依赖于前一个时间步的结果
▪ 这会导致训练时间较长
▪ 随着序列长度增加，扩展性较差（难以处理很长的序列）

![](image/Pasted%20image%2020260220083357.png)


## 2.7 SCALING LAWS FOR NEURAL LANGUAGE MODELS



Model Size Scaling
Model performance increases with the number of parameters, following a power-law relationship, but eventually plateaus once it reaches an underfitting regime.
随着参数数量的增加，模型性能会提升，并通常遵循一种 幂律关系（power-law relationship）。
但当进入欠拟合（underfitting）区域后，性能提升会逐渐趋于平缓（plateau）。

Data Scaling
Adding more data enhances performance up to a saturation point, beyond which the model risks overfitting unless its size is increased
增加训练数据通常会提升模型性能。
但当达到某个饱和点（saturation point）后，如果不同时增加模型规模，模型可能会出现过拟合（overfitting）风险。

Compute-Optimal Scaling
An optimal frontier exists where compute, model size, and data are balanced, deviating from it results in inefficient resource use.
存在一条最优前沿（optimal frontier），在这条曲线上，计算资源（compute）、模型规模（model size）和数据量是平衡的。
偏离这条最优平衡点会导致计算资源使用效率低下。



As you increase the size of the model (number of parameters) and the amount of training data, the model's ability to predict language gets better in a predictable way.
![](image/Pasted%20image%2020260220083629.png)

![](image/Pasted%20image%2020260220083752.png)


# 3 TRANSFORMER 


Transformer
Encoder encodes input into contextualized embeddings while decoder generates output based on contextualized embeddings and partial output. Initially developed for language translation.
Key innovation: Multi-head attention layers.
Variants:
T5: Encoder and decoder modules for text translation
BERT: Encoder module for text classification GPT: Decoder module for text generation

![](image/Pasted%20image%2020260220084656.png)


---

![](image/Pasted%20image%2020260220084832.png)

1 
Input text to be translated

2 
Input text is prepared for the encoder

3 
Encoder produces text encodings from complete input text

4 
Embeddings of the input sequence

---



5 Partial output text

6 Input text is prepared for the decoder

7 Decoder generates the translated text

8 Preparation of output

Translated text

----

![](image/Pasted%20image%2020260220085113.png)

![](image/Pasted%20image%2020260220085123.png)


# 4 TOKENIZER 

![](image/Pasted%20image%2020260220085151.png)

Tokenization
Breaks text into a sequence of units a model can learn from

Vocabulary Creation
Model needs to deal with numbers instead of arbitrary units to compute efficiently

---
What if a user during application enters a word that was not contained in the training data?
![](image/Pasted%20image%2020260220085230.png)

----
Tokenizer for GPT-3 does not use <|unk|>. Instead, it breaks down words into subwords.
![](image/Pasted%20image%2020260220085311.png)


---

Tokenizer for GPT-[2-4] apply BPE
Calculate probabilities of byte- pairs. Concatenate highly probable byte-pairs. Stop if target vocabulary size reached.
Byte-pair Encoding (BPE)

![](image/Pasted%20image%2020260220085324.png)


# 5 EMBEDDINGS 

![](image/Pasted%20image%2020260220091756.png)

![](image/Pasted%20image%2020260220091832.png)


Key Concept
▪ ==Each word is a continuous vector in an n-dimensional vector space, denoted as embedding==
▪ "Close" embeddings represent semantically similar concepts
▪ Goal: Learn optimal embeddings using neural networks


Used in ...
▪ Distributed word feature vector *1 ▪ Word2Vec *2
▪ BERT, GPT
▪ Vector Databases

![](image/Pasted%20image%2020260220090609.png)


## 5.1 IN 3-DIMENSIONAL VECTOR SPACE

The closer the more aligned to each other


![](image/Pasted%20image%2020260220090836.png)

---

dot product: Measures how well both embeddings align to each other
![](image/Pasted%20image%2020260220090856.png)


![](image/Pasted%20image%2020260220090926.png)

![](image/Pasted%20image%2020260220090944.png)

## 5.2 EMBEDDINGS MATRIX


![](image/Pasted%20image%2020260220091809.png)



![](image/Pasted%20image%2020260220091001.png)

## 5.3 positional encoding:


Problem
▪ Transformers have no recurrence or convolution, so they don't know the word order within a given word sequence
▪ "The dog chased the cat" = "The cat chased the dog" 

Solution
▪ Encode the position of a word within the sequence into the word embedding itself.
𝑒റ + 𝑝റ = 𝑒റ
𝑡𝑜𝑘𝑒𝑛 𝑡𝑜𝑘𝑒𝑛_𝑖𝑛𝑝𝑢𝑡
▪ Encode either absolute*1 or relative*2 position and either fixed or learned absolute or relative position
▪ Result: "The dog chased the cat" ≠ "The cat chased the dog"

![](image/Pasted%20image%2020260220091031.png)



----

Requirements
What properties do the positional encoding vectors need to exhibit?
▪ Unique encoding for each position
▪ Linear relation between two encoded positions
▪ Generalize to longer sequences than those in training
▪ Generated by a deterministic process the model can learn
▪ Extensible to multiple dimensions
▪ Value range should be somewhere between -1 and 1 for numerical stability
▪ Only small, smooth, continuous and predictable changes between two adjacent positional encoding vectors

▪ 每一个位置都必须具有唯一的编码（Unique encoding for each position）
▪ 不同位置之间应当存在线性关系（Linear relation between two encoded positions）
▪ 能够泛化到比训练时更长的序列（Generalize to longer sequences than those in training）
▪ 由模型可以学习的确定性过程生成（Generated by a deterministic process the model can learn）
▪ 能够扩展到多维空间（Extensible to multiple dimensions）
▪ 数值范围应在 -1 到 1 之间，以保证数值稳定性（Numerical stability）
▪ 相邻两个位置编码之间应只有小幅、平滑、连续且可预测的变化（Small, smooth, continuous and

----
1. be unique
2. show the realtionship of each of position. The distance of two differnt position 



A position vector for each position in the sequence. Does not change during learning. Position vector has same dimension as the embeddings.

![[images/Pasted image 20251113104433.png]]


![](image/Pasted%20image%2020260220091001.png)

这两个 matrix  垂直方向的维度都是 d_model = embeddings diemenesion, 一列 对应一个 embedding in one position 

----





## 5.4 LAYER NORM

Layer Normalization
▪ Normalizes a single input vector across its features independent of other input vectors.
▪==Goal:𝜇≈0 and𝜎≈1 among features==
▪ ==Stabilizes training by keeping activations well-scaled across layers==
▪ In other words,== layer normalization keeps a token's internal representation balanced. No feature dominates and learning stays steady.==

▪ 对单个输入向量在其特征维度上进行归一化，与其他输入向量无关。
▪ 目标：使各特征的

均值（μ）≈ 0

标准差（σ）≈ 1

▪ 通过在各层之间保持激活值（activations）处于良好的尺度范围，从而稳定训练过程。

▪ 换句话说，层归一化可以保持一个 token 的内部表示是平衡的——
不会有某个特征占据主导地位，训练过程也会更加稳定。


![](image/Pasted%20image%2020260220092128.png)

Well balanced


# 6 ATTENTION 

![](image/Pasted%20image%2020260220095546.png)


The meaning of syntactically similar words may vary depending on the context of use.

![](image/Pasted%20image%2020260220092439.png)


what kind of words influeces the current of words right now 
![[images/Pasted image 20251113105417.png]]

----


==Attention layer transforms a token embedding into a contextualized token embedding==

![](image/Pasted%20image%2020260220092602.png)


## 6.1 SINGLE-HEAD SELF-ATTENTION


### 6.1.1 Query and key 


==重要 ==
Query = what I search for
Key = how I match others
Value = what I contribute

Query： Is there a word in front of me that helps me to find my real meaning?
The query is what a word is looking for.
It asks: "Which other words are relevant to me?"


Key： I am a word that could give a word following me its meaning.
The key is what a word offers.
It says: "If you are looking for something like this, I might be important."


The value is the actual information a word provides.
After comparing queries and keys to compute attention scores, the model combines the values to produce the final representation.



![[images/Pasted image 20251113110115.png]]


----

Check how much related they are via dot product
Masking: we only use the post token influcence the cureent token  
![[images/Pasted image 20251113110625.png]]

![](image/Pasted%20image%2020260220093123.png)


---
### 6.1.2 Attention score matrix and softmax()
Attention score matrix
Calculate softmax() for each column
![](image/Pasted%20image%2020260220093147.png)

![](image/Pasted%20image%2020260220093155.png)



### 6.1.3 Masking
---
Masking before softmax() so that later words never influence earlier words (apply causal mask)

![](image/Pasted%20image%2020260220093237.png)


---

### 6.1.4 
Scalability of Transformer
▪ Limited by context windows size because the attention is calculated by dot-multiplying all key vectors with all query vectors.
▪ Attention complexity increases exponentially with context windows size
▪ 受限于上下文窗口大小（context window size），因为注意力是通过将所有 key 向量与所有 query 向量进行点乘计算得到的。
▪ 随着上下文窗口大小增加，注意力计算的复杂度呈指数级增长。（更准确来说是平方级增长 O(n²)）


Alternative Attention Mechanisms
▪ Sparse attention mechanism ▪ Blockwise attention
▪ Linformer
▪ Reformer
▪ Ring attention
▪ Longformer
▪ Adaptive attention spa

▪ 稀疏注意力机制（Sparse Attention Mechanism）
▪ 分块注意力（Blockwise Attention）
▪ Linformer
▪ Reformer
▪ Ring Attention
▪ Longformer
▪ 自适应稀疏注意力（Adaptive Attention Sparsity）


![](image/Pasted%20image%2020260220093420.png)



### 6.1.5 value

The value is the actual information a word provides.
After comparing queries and keys to compute attention scores, the model combines the values to produce the final representation.


The value is the actual information a word provides.
After comparing queries and keys to compute attention scores, the model combines the values to produce the final representation.

![](image/Pasted%20image%2020260220093739.png)

### 6.1.6 最后的 attention score matrix 


![](image/Pasted%20image%2020260220093831.png)


Influence from any earlier word in the context window
Attention calculates delta and adds it to embedding

![](image/Pasted%20image%2020260220093952.png)


## 6.2 MULTI-HEAD SELF-ATTENTION

Concept
▪ Calculate multiple attention weights matrixes for different types of contextual relationships.
▪ Each attention head looks through a different lens 
▪ Each attention head determines a single attention weight matrix using own 𝑊 , 𝑊 and 𝑊 matrices 𝑄𝐾𝑉
All resulting ∆𝑒റh with h = h𝑒𝑎𝑑 are added to the original 𝑒റ 𝑡𝑜𝑘𝑒𝑛 𝑡𝑜𝑘𝑒𝑛

GPT-3 used 96 attention heads per attention layer and 96 attention layers in total

▪ 计算多个注意力权重矩阵，以捕捉不同类型的上下文关系。
▪ 每个注意力头都从不同的"视角"理解输入信息。
▪ 每个注意力头使用各自独立的投影矩阵 Wᵠ、Wᵏ 和 Wᵛ 来计算自己的注意力权重矩阵。
▪ 每个注意力头都会生成一个表示更新 Δe_token^h。

▪ 所有头产生的 Δe_token^h（h 表示不同的 attention head）都会被合并，并通过残差连接加到原始的 token 表示 e_token 上。


![](image/Pasted%20image%2020260220095229.png)




![](image/Pasted%20image%2020260220094909.png)


## 6.3 CROSS-ATTENTION

Concept
▪ Find the attention weights matrix of one token sequence to another token sequence
▪ ==It lets the decoder look at and use important information form the encoder's output when generating each word==
▪ Technically, it lets the decoder compute context- dependent representations by using 
- its current hidden states (what has been generated so far) as queries and 
- the encoder's output embeddings as keys and values

As GPT-3 is a decoder-only model it makes no use of a cross-attention layer!


▪ 计算一个 token 序列对另一个 token 序列的注意力权重矩阵。
▪ 它允许解码器（decoder）在生成每个词时，查看并利用来自编码器（encoder）输出的重要信息。
▪ 从技术上讲，解码器使用其当前的隐藏状态（即到目前为止已经生成的内容）作为 Query， 而将编码器的输出嵌入作为 Key 和 Value，
从而计算出依赖于上下文的表示（context-dependent representations）。

![](image/Pasted%20image%2020260220095314.png)
# 7 MLP  MULTI-LAYER PERCEPTRON


![](image/Pasted%20image%2020260220100847.png)



Multi-layer Perceptron
▪ A multi-layer perceptron (MLP) is a type of ANN made up of multiple layers of neurons, where each layer is fully connected to the next.
▪ Simplest form of a feed-forward neural network
▪ Consists of input layer, one or more hidden layers and one output layer
▪ Each neuron performs an activation function
▪ General function approximator by using non-linear activations

▪ 多层感知机（MLP）是一种人工神经网络（ANN），由多个神经元层组成，每一层都与下一层全连接（fully connected）。

▪ 它是最简单形式的前馈神经网络（feed-forward neural network）。

▪ 由输入层（input layer）、一个或多个隐藏层（hidden layers）以及输出层（output layer）组成。

▪ 每个神经元都会执行一个激活函数（activation function）。

▪ 通过使用非线性激活函数，MLP 可以作为一个通用函数逼近器（general function approximator）。



![](image/Pasted%20image%2020260220100323.png)

---

MLP in the Transformer
▪ Each contextualized embedding from the attention layer runs independently through the MLP (same MLP with same 𝑊𝑙𝑠 and 𝑏𝑙𝑠)
▪ While the attention layer refined an embedding by looking onto its context
in the input sequence, the MLP refines each contextualized embedding 𝑒റ′
Feed Forward
independently of the other tokens in the input sequence ▪ MLP characteristics:

作用的是 Enrich with more information

▪ 位置独立（position-wise）计算
▪ 对每个 token 使用相同的参数（参数共享）
▪ 通常包含两层线性变换，中间加非线性激活函数（如 ReLU 或 GELU）
▪ 扩展维度后再压缩（通常先升维再降维）
▪ 增强模型的非线性表达能力


![](image/Pasted%20image%2020260220100411.png)

---

![](image/Pasted%20image%2020260220100514.png)


## 7.1 FORWARD PASS

1  Interpret as encoding the characteristics of a bank with influence on the inflation

Interpret as encoding the characteristics of a central bank
![](image/Pasted%20image%2020260220100545.png)


![](image/Pasted%20image%2020260220100602.png)


## 7.2 ADDING THE DELTA

![](image/Pasted%20image%2020260220100725.png)


## 7.3 PARAMETERS

![](image/Pasted%20image%2020260220100739.png)

# 8 Output 



## 8.1 LINEAR PROJECTION


Concept
▪ ==Find the "similarities" of the last embedding of the sequence with all words of the vocabulary==
▪ ==Mathematically, calculate the dot product of the last embedding of the sequence with every word of the vocabulary==
▪ Result is a list of logits (scores)

GPT-3 uses an unembedding matrix 𝑊 for the final linear
𝑢𝑛𝑒𝑚𝑏𝑒𝑑
projection instead of 𝑊 𝑒𝑚𝑏𝑒𝑑

▪ 计算序列中最后一个嵌入向量与整个词汇表中所有词向量之间的"相似度"。
▪ 从数学上讲：
将序列最后一个嵌入向量与词汇表中每一个词向量进行**点积（dot product）**计算。
▪ 结果是一个 logits（分数）列表，每个分数对应词汇表中的一个词。



![](image/Pasted%20image%2020260220101100.png)




## 8.2 SOFTMAX

![](image/Pasted%20image%2020260220101453.png)


Concept
▪ Transform the logits into probabilities with the softmax function
▪ Parametrize with temperature to influence the randomness of the result

Tempurature 
▪ Controls randomness and creativity during generation

= 1: No change. The model behaves as trained.
< 1: Increases relative differences. Probabilities become sharper and more confident.
  \> 1: Reduces relative differences. Probabilities become flatter and more uniform.


![](image/Pasted%20image%2020260220101556.png)



## 8.3 ALTERNATIVE SAMPLING STRATEGIES


Top-k Sampling
▪ Restricts which tokens can be chosen
▪ Step 1: Sort probabilities in a decreasing order
▪ Step 2: Keep only the k-highest and remove all other ▪ Step 3: Renormalize remaining probabilities
▪ Typical k = 20-100

---

Top-p (nucleus) Sampling
▪ Adaptive version of top-k sampling
▪ Step 1: Sort probabilities in a decreasing order
▪ Step 2: Choose the smallest subset of tokens whose cumulative probability is ≥ a given probability 𝑝, e.g., 0,9
▪ Step 3: Renormalize within the subset
▪ Typical 𝑝 ≈ 0,9-0,95

---

Entropy-based Sampling (Typical Sampling)
▪ Use the entropy of the distribution to decide which tokens to keep ▪ Keep tokens with log-probabilities of within a certain "typicality"
band around the expected surprise 
▪ Reduces degenerate repetition
Typical Sampling 不是只选最高概率的词，而是选择"既不太常见也不太罕见"的典型词，从而让生成更自然、更多样。

![](image/Pasted%20image%2020260220101649.png)

3  Presence and Frequency Penalties
▪ Adjust logits before softmax to discourage overused tokens
▪ To reduce degenerative loops like "The cat sat on the mat on the
mat on the mat..."

![](image/Pasted%20image%2020260220101759.png)



4 Logit Biasing / Masking
▪ Logit Biasing: Manually increase or decrease scores of certain tokens
▪ Logit Masking: Set tokens that should be impossible to −∞ to exclude them entirely

▪ Logit Biasing（Logit 偏置）：
手动提高或降低某些 token 的分数（logits），从而影响它们被选中的概率。
▪ Logit Masking（Logit 屏蔽）：
将不允许出现的 token 的分数设置为 −∞，从而完全排除它们。


8 Greedy Decoding
▪ Always take the token with the highest probability ▪ It makes the locally best choice
▪ Fast and deterministic
▪ Best for classification, code and math
▪ Not good for creative or open-ended generation

▪ 始终选择概率最高的 token。
▪ 做出当前步骤的局部最优选择（locally best choice）。
▪ 速度快，且结果是确定性的（deterministic）。
▪ 适用于分类、代码生成和数学推理等任务。
▪ 不适合创造性或开放式文本生成，因为容易缺乏多样性。



9  Beam Search
▪ Exploring several top partial hypotheses in parallel
▪ Argmax maximizes local probabilities while beam search tries to
find the global sequence probability
▪ Keeps top 𝐵 (beam width) candidate sequences at every decoding step
▪ Repeats until EOS or max length is exceeded
▪ Output the sequence with the highest total log-probability
▪ More deterministic but lesser creative
▪ Used for translation, speech recognition and summarization
▪ Computationally expensive

![](image/Pasted%20image%2020260220101902.png)




![[images/Pasted image 20251120104430.png]]


the final embedding (it accumalte all the meaning of the prvious words). this embedding is a konzept
demensinal of embidding vectorr:   demsionmen of d_model  , d_model is the matrix 

linaer projection:  understanding the embeding,  tanslate it into all possibele konzept (each konzept hase a score value)

softmax: scale it into `[0,1]`

sampling: strategy how to generate the disturtion , sue this distrition help use to generate the new words 



## 8.4 temperature  in output processing 


你说的 “tempatur” 多半是 temperature（温度），它不是 Transformer 架构本身的组成部分，而是 在 Transformer 生成文本（如 GPT、T5 Decoder 等）时用于控制输出随机性的一个超参数


![[images/Pasted image 20251120104946.png]]





# 9 Experiment 

![[images/Pasted image 20251120104533.png]]












