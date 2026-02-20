

# 1 PRE-TRAINING 

![](image/Pasted%20image%2020260219180700.png)


![](image/Pasted%20image%2020260219181019.png)


----

## 1.1 TERMS

Step / Iteration
A single update of the model's parameters

Batch
One or multiple sequences of tokens

Batch size
LLM practice: Total number of tokens within a batch 
ML: Number of sequences per batch
max length of sequence  can be allowed    is the the size of content window 
content size  is denfind by LLM,   key value matrix 决定了 content size 
each sequence in a batch is in the same size 



Training Data
Corpus of text data

Epoch
A single training pass through the training data


Sequence Length
Number of the tokens of a sequence

Steps per Epoch
Size of training data divided by the batch size

Loss
A numerical measure of how much the model's predictions differ from the expected answers (ground truth)

Checkpoint
Saved model state



----

## 1.2 CONSIDERATIONS

Gridient vanishing:  the input hat no influence on the last layer  
Gridient Explode: a little change in input has huge impact on the output 


GTP-3 175B used a batch size of 3.2 million tokens with about 1.562 sequences per batch

Define the Scope and Resources
▪ General LLM, domain-specific LLM or a code LLM ▪ Number of parameters
▪ Technical constraints
▪ Budget constraints
▪ Context length, tokenizer, parameter precision ▪ Architecture: Decoder-only, depth, width, etc.
通用 LLM、领域专用 LLM 或代码专用 LLM
参数数量
技术约束
预算约束
上下文长度、分词器（tokenizer）、参数精度
架构设计：仅解码器（Decoder-only）、网络深度（depth）、宽度（width）等

Training Data
Public data, licensed data, synthetic data
Policy-based filtering, cleaning, deduplication
Train tokenizer, fix vocabulary size, special tokens
"Batchify" the data: pack multiple sequences into fixed-length blocks to maximize GPU utilization
Sharding
公开数据、授权数据、合成数据
基于策略的数据过滤、清洗、去重
训练分词器，确定词表大小和特殊 token
对数据进行"批处理化（batchify）"：
将多个序列打包成固定长度的块，以最大化 GPU 利用率
数据分片（sharding）


Training Setup
▪ Learning-rate schedule: How much the LLM is adjusted at each iteration?
▪ Optimizer: How the parameters are adjusted at each iteration
▪ Dropout rate: Randomly ignore some activations to prevent overfitting
▪ Worker nodes, slicing of training data
▪ Types of parallelism, types of distribution
▪ Metrics: training loss, validation loss, hardware utilization, throughput, etc.
▪ Error handling: Exploding gradients, etc. 

学习率调度（Learning-rate schedule）：
每次迭代中对 LLM 参数的调整幅度

优化器（Optimizer）：
每次迭代中参数如何更新

Dropout 比率：
随机忽略部分神经元激活，以防止过拟合

工作节点数量、训练数据切分方式

并行方式与分布式训练类型

评估指标：
训练损失（training loss）、验证损失（validation loss）、硬件利用率、吞吐量等

错误处理：
如梯度爆炸（exploding gradients）等问题


Post-training
▪ Supervised fine-tuning
▪ Safety and policy tuning
▪ Test jailbreaks, prompt-injection and hallucination-prone topics

监督微调（Supervised fine-tuning）
安全与策略调优（Safety and policy tuning）
测试越狱攻击（jailbreak）、提示注入（prompt injection）以及容易产生幻觉的话题



## 1.3 TRAINING ITERATION STEP BY STEP

1 Data Loading and Preparation
▪ Create next batch and shift for prediction
▪ If distributed pre-training, create mini batches for each GPU.
构造下一批（batch）数据，并进行"错位/右移（shift）"以用于预测下一个 token。
如果是分布式预训练，则为每张 GPU 分配并生成各自的 mini-batch。

2  Forward Pass
▪ Forwards pass (inference) with batch as input 
 As explained in Chapter 2
 将一个 batch 作为输入进行前向计算（推理 / inference）。
如第 2 章所解释的那样。


--- 

3  Loss Computation
Determine cross-entropy loss 𝐿 (also known as negative log- likelihood) as a measure of how well a predicted probability distribution matches the true distribution of the next token.
![](image/Pasted%20image%2020260219181627.png)

▪ If model assigns high probability to the correct token, then loss is small. Otherwise, loss is large.
▪ Cross entropy is a generalized, expanded form of Log-Loss. (Binary) Log-loss is used for two-class classification tasks while cross entropy is used for the multi-class classification task. 
An LLM is – basically - a multi-class classification model with 𝑑_𝑣𝑜𝑐𝑎𝑏 classes.

计算交叉熵损失 
L（也称为负对数似然，negative log-likelihood），用来衡量模型预测的"下一个 token 的概率分布"与真实分布的匹配程度。

如果模型给正确 token 分配了很高的概率，那么损失就很小；否则损失就很大。
交叉熵是 Log-Loss 的更一般、更扩展的形式。
（二分类）Log-loss 用于两类分类任务
交叉熵用于多分类任务

一个 LLM 从本质上来说就是一个多分类模型，其类别数为 （即词表大小）。

---

4 Backward Pass
Compute model parameter "changes" starting from the back
Gradient is how much a tiny change of the parameter changes the loss.
Gradients gives an optimizer the information of how to move the parameters to reduce future loss
As each layer is a function 𝑓(𝑒റ), the transformer can be interpreted as a composition of functions:

![](image/Pasted%20image%2020260219205140.png)


What operations happen to an embedding within one transformer block (simplified):

![](image/Pasted%20image%2020260219205152.png)

----

5  Optimizer step
==gradient vanishing:  the input hat no influence on the last layer  ==
==gradient Explode: a little change in input has huge impact on the output ==
==gradient: find the direct where the loss change at most ==

The gradient information computed in step 4 (backpropagation) is used in this step to update the model parameters
The gradient give the direction of change that increases the loss 
Potential optimizer: Stochastic Gradient Descent (SGD) or Adam

![](image/Pasted%20image%2020260219205846.png)

![](image/Pasted%20image%2020260219205933.png)

▪ Adam (Adaptive Momentum Estimation) is used by most GPT models
▪ Adam's key idea is a) to keep the running average of recent gradient and b) scale each parameter's step size based on how large or small its recent gradients are
▪ ==SDG computes a global step size while Adam computes a parameter-specific adaptive step size==
▪ SDG remembers only one previous update via momentum while Adam knows variance of past gradients


---

6  Logging and Checkpointing
▪ Logging: to keep track of what happens during the training ▪ Monitor learning progress (loss, accuracy, etc.)
▪ Detect problems (e.g., exploding loss, vanishing gradients) ▪ Compare runs with different hyperparameters
▪ Decide when to stop or adjust training
▪ Checkpointing: to save model and optimizer states ▪ Resume training after interruption
▪ Roll back if training diverges


## 1.4 METRICS

Cross-entropy
Measures the dissimilarity between the true distribution and the predicted distribution [for training]
![](image/Pasted%20image%2020260219210218.png)



Perplexity
Measures the uncertainty of a probabilistic model [for evaluation]
![](image/Pasted%20image%2020260219210305.png)


Next-token Accuracy (Top-1 Accuracy)
Measures ==how often ==in a test the most probable token matches the true token [for evaluation]
衡量在测试中，模型预测的最高概率 token是否与真实 token 匹配的频率。
【用于评估】


Top-k Accuracy
Generalized form of next-token accuracy. Measures how often in a test the true token was amongst the k most probable tokens. Reasonable if several tokens make sense, e.g., as with synonyms or punctuation. [for evaluation]
衡量在测试中，真实 token 是否出现在模型预测的 前 k 个最高概率 token 之中。

当多个 token 都是合理答案时（例如同义词或标点符号），该指标更合理。
【用于评估】


Expected Calibration Error (ECE)
==Measures how far confidence is from reality. ==Confidence is the highest probability of an output distribution. ECE (simplified) is the difference between average confidence and average accuracy. If the model is on average 80% confident, is it correct about 80% of the time? [for evaluation]

衡量模型"置信度"与"真实准确率"之间的差距。
置信度：输出概率分布中的最大概率值
ECE（简化理解）：平均置信度与平均准确率之间的差值

例如：
如果模型平均置信度为 80%，它是否真的在 80% 的情况下预测正确？



Brier Score
Calculate element-wise the difference between probability distribution vector and on-hot vector for the true label. Sum the squared differences. Brier score is higher if confidence is high, but label is wrong, and lower if confidence is low and label is wrong. It quantifies how close a model's predicted probabilities are to the true outcomes [for evaluation]

计算方法：
- 对每个类别，计算预测概率分布向量与真实标签的 one-hot 向量之间的差值
- 将差值平方后求和

特点：
- 如果模型置信度很高但预测错误，Brier Score 会较高
- 如果模型置信度较低且预测错误，Brier Score 较低

该指标用于衡量模型预测概率与真实结果之间的接近程度。

# 2 Fine Tunning


## 2.1 Intro

Result of Pre-Training
▪ LLMs became aware of universal linguistic patterns (grammar, syntax, vocabulary, idiomatic, etc.)
▪ LLMs own world knowledge (facts, cultural and scientific knowledge, etc.)
▪ LLMs exhibit emergent capabilities 


Why Fine-tuning?
▪ Tailoring an LLM to specific tasks
▪ Fine-tuned LLMs can be computationally more efficient 
▪ Incorporate domain knowledge
▪ Improved instruction following
▪ Alignment with political or institutional guidelines
▪ Compliment with legal standards
▪ To act safe and polite
▪ Reduce complexity of prompt engineering
▪ Continuous learning

![](image/Pasted%20image%2020260219210946.png)


Fine-tuning Strategies
▪ ==Domain-specific Fine-Tuning. ==LLM should learn the domain specific language and knowledge. Exemplary domains are legal and medical.
▪ ==Instruction fine-tuning (task-specific fine-tuning).== Adapts LLMs for specific downstream tasks such as text summarization, code generation, classification, Q&Q, chat, etc.

Learning Techniques for Fine-tuning
▪ Self-supervised Learning 
▪ Supervised Learning
▪ Reinforcement Learning

Major Challenges
▪ Lack of training data
▪ Limit of Hardware Resources ▪ Catastrophic forgetting


## 2.2 INSTRUCTION FINE-TUNING


![](image/Pasted%20image%2020260219211535.png)


### 2.2.1 INSTRUCTION FINE-TUNING: STEP BY STEP

![](image/Pasted%20image%2020260219211637.png)

![](image/Pasted%20image%2020260219211645.png)

![](image/Pasted%20image%2020260219211910.png)


![](image/Pasted%20image%2020260219211920.png)

Full instruction masking is the widely used masking technique in instruction fine-tuning. But other approaches exist*
![](image/Pasted%20image%2020260219211933.png)


Average loss for all tokens of the sample
![](image/Pasted%20image%2020260219212214.png)


![](image/Pasted%20image%2020260219212321.png)

### 2.2.2 INSTRUCTION DATA SETS


Popular Publicly Available Datasets
▪ Alpaca dataset: Pairs of instructions and ideal/to-be-expected responses. Sometime also context in addition.
▪ Flan Collection (manual)
▪ Databricks-Dolly-15k (manual)
▪ OpenOrca
▪ Platypus
▪ OpenHermes
▪ Falcon
▪ ChatAlpaca
▪ Auto CoT


![](image/Pasted%20image%2020260219212537.png)

![](image/Pasted%20image%2020260219212644.png)



## 2.3 MULTI-TURN FINE-TUNING

Concept
▪ Multi-turn fine-tuning is the process of teaching an LLM to ==understand the concept of turn-by-turn conversation==
▪ The goal is that an LLM should maintain context across multiple exchanges while adhering to a specific conversation patterns
▪ Important for a chat client or, more generally, any kind of multi- hop task completion
==Highly relevant for AI agents because of multi-turn function calling, where an AI agent is instructed to call tools in a preferred sequence and make decisions based on the intermediate results.==

单轮微调让模型"会回答问题"
多轮微调让模型"会持续对话并完成复杂任务"

多轮微调（Multi-turn fine-tuning） 是指训练 LLM 去理解"逐轮对话"的概念。

目标是让 LLM 能够：
在多次对话往返中保持上下文连续性
同时遵循特定的对话模式
这对于聊天客户端（chat client）或更广义上的多跳任务（multi-hop task）完成非常重要。

对于 AI Agent 来说尤为关键，因为：
Agent 需要进行多轮函数调用（multi-turn function calling）
模型需要按照指定的顺序调用工具
并根据中间结果做出决策

![](image/Pasted%20image%2020260219212908.png)

----



## 2.4 METHODS

Task- specific Fine-tuning
Instruction Fine-tuning
Domain Fine-tuning
Policy Fine-tuning

![](image/Pasted%20image%2020260219213054.png)

## 2.5 PARAMETER EFFICIENT FINE-TUNING

Concept
▪ Goal is to minimize memory footprint, improve storage efficiency or to add modularity
▪ Option 1: Add new parameters (additive)
▪ Option 2: Fine-tune subset of existing parameters (selective)
▪ Option 3: Reparametrize with new representation

Adapters (additive)
▪ Add new trainable layer to the model architecture 

Soft Prompts
"Fine-tuning of input" for inference
Focuses on enriching the input with learned soft prompts (non- human readable) to steer the inference process

![](image/Pasted%20image%2020260219213132.png)



## 2.6 LOW RANK ADAPTATION (LORA)

Concept
▪ Track changes to the model's weights and freeze original weights
▪ ==Split the changes matrix into two submatrices (LoRA matrices) with low rank ==(known as matrix decomposition)
▪ ==Each backward pass only affects the LoRA matrices==
▪ Precision is sacrificed for fine-tuning efficiency
▪ ==Rank determines target precision to be sacrificed==
▪ LoRA is more efficient, faster and less prone to catastrophic forgetting since original model weights are kept

跟踪模型权重的变化，同时冻结原始权重
将"权重变化矩阵"拆分为两个低秩子矩阵（称为 LoRA 矩阵），这是一种矩阵分解方法
每一次反向传播（backward pass）只会更新 LoRA 矩阵
为了提升微调效率，会牺牲一定的精度
Rank（秩）决定了可接受的精度损失程度
LoRA 更高效、更快，并且更不容易出现灾难性遗忘（catastrophic forgetting），因为原始模型权重保持不变


Low Rank
▪ To teach for simple downstream tasks
适用于训练简单的下游任务

High Rank
▪ To teach for complex downstream tasks
▪ For incorporating contradicting or new behavior
适用于引入相互矛盾或全新的行为模式

![](image/Pasted%20image%2020260219213243.png)

---

Concept
▪ LoRA module = LoRA matrices (after fine-tuning)
▪ LoRA matrices are additive in nature
▪ Multiple LoRA modules can be loaded into memory
▪ Route prompts to ideal LoRA module-adapted version of the basic model in the tree


![](image/Pasted%20image%2020260219213443.png)

![](image/Pasted%20image%2020260219213509.png)




## 2.7 LIMITATIONS OF INSTRUCTION FINE-TUNING

Supervised fine-tuning can imitate good behavior, but it cannot evaluate, prefer, or improve behavior.



![](image/Pasted%20image%2020260219213548.png)


## 2.8 POLICY FINE-TUNING：  ALIGNMENT WITH HUMAN PREFERENCES:
REINFORCEMENT LEARNING (RL): THE BASICS


![](image/Pasted%20image%2020260219213724.png)

RL FOR LLMS
![](image/Pasted%20image%2020260219213743.png)



### 2.8.1 RL Optimize Policy 

![[image/Pasted image 20251204104600.png]]

Optimize a policy against the reward model using PPO
Step 3.1: Generation
LLM generates multiple outputs in response to given instruction

Step 3.2: Rewarding
The reward model assign each output a scalar score (of how well it align with human preferences)

Step 3.3: Compute Expected Reward
Compute what reward the model would expect for the generated output.

Step 3.4: Compute Advantages
==Compute what parts of the answer were good and what parts were bad by taking the difference between expected and real reward into consideration.==

Step 3.5: Compute Changes
Compute how much the policy must change without deviating too much from the original policy (from Step 1).

Step 3.6: Update Policy
Model weights are updated




![[image/Pasted image 20251204105651.png]]


---

PROXIMAL POLICY OPTIMIZATION
近端的；近源的；


Decide Which Way to Adjust
▪ Compare each response's reward to the model's expected reward (estimated by the value head).
▪ The difference is the advantage: how much better or worse this answer was than average.
▪ The advantage dictates the direction and strength of adjustment 

Small, Safe Steps (Clipping)
▪ Measure how much the new model's probability of each token differs from the previous version's (old policy)
▪ If an update would make a token more than 20% more or less likely, it get clipped

Stay Close to Reference Model
Compute the Kullback-Leibler (KL) divergence between current model and reference model
Add penalty to the loss or reduce learning rate if too divergent


Repeat in Mini-Batches
▪ Apply gradient updates 
▪ Iterate over training data

Stop When Stable
▪ Stop if reward scores stabilize, human evaluators stop seeing improvement or when KL divergence starts to rise too fast (model drifting too far)



决定如何调整（Decide Which Way to Adjust）
* 将每个回答的奖励（reward）与模型的期望奖励（由 value head 估计）进行比较。
* 两者之间的差值称为 **优势（advantage）**：表示该回答比平均水平好多少或差多少。
* 优势值决定了参数调整的方向和强度。

---

小而安全的更新步骤（Clipping）
* 衡量新模型中个 token 的概率与旧版本模型（旧策略，old policy）相比发生了多大变化。
* 如果某次更新使某个 token 的概率增加或减少超过 20%，则会被"裁剪（clipped）"。

👉 目的是避免模型参数发生过大的剧烈变化。

---

保持接近参考模型（Stay Close to Reference Model）
* 计算当前模型与参考模型之间的 **KL 散度（Kullback-Leibler Divergence）**。
* 如果差异过大：
  * 在损失函数中加入惩罚项
  * 或降低学习率

👉 防止模型偏离参考模型太远。

---

以 Mini-Batch 形式重复（Repeat in Mini-Batches）
* 执行梯度更新
* 在训练数据上反复迭代

---

在稳定时停止（Stop When Stable）
在以下情况停止训练：

* 奖励分数趋于稳定
* 人类评估者不再观察到明显改进
* KL 散度开始快速上升（说明模型偏离过远）

---

整体来看，这描述的是 **PPO（Proximal Policy Optimization）在 RLHF 中的训练流程核心思想**：

* 用 advantage 决定更新方向
* 用 clipping 控制更新幅度
* 用 KL penalty 控制模型漂移
* 在稳定时停止训练

如果你愿意，我可以帮你画一个"RLHF + PPO 训练流程逻辑图"，把 reward model、policy model、reference model 三者关系串起来。



## 2.9 DIRECT SEQUENCE OPTIMIZATION (DPO)

RLHF 是"训练一个奖励模型 + 用 PPO 优化策略"
DPO 是"直接用偏好数据做对比学习式优化"


Concept
▪ Derived from RL principles. DPO is not RL.
▪ Developed as a simplification of RLHF
▪ Instead of using a reward model, DPO makes the preferred output more likely next time
▪ Three main objectives: 1) Increase the probability of the preferred answer, 2) decrease the probability of the non-preferred answer and 3) do not drift too far from the original model
来源于强化学习（RL）的原理，但 DPO 本身并不是强化学习。
作为 RLHF 的一种简化方法而提出。
不再使用奖励模型（reward model），而是直接让"更受偏好的输出"在下次出现时具有更高的概率。

DPO 的三个主要目标：
提高被偏好答案（preferred answer）的概率
降低非偏好答案（non-preferred answer）的概率
不要偏离原始模型太远


Properties
▪ No trial and error
▪ No reward model
▪ No RL algorithm such as PPO
▪ No simulation of action and rewards required
▪ Simpler, cheaper (fewer GPUs) and more transparent

![](image/Pasted%20image%2020260219214748.png)


# 3 RL WITH AI FEEDBACK (RLAIF)

![](image/Pasted%20image%2020260219214837.png)

# 4 Distillation
ˈdɪstɪleɪt

## 4.1 INTRO

Concept
▪ ==Transfer of knowledge from a "teacher" ANN to a "student" ANN==
▪ Originally introduced by Hinten et. al in 2015*
▪ Student model is trained to match the teacher's output as close as possible
▪ Student model has lesser complexity and thus lesser computational requirements without loosing too much performance in comparison to the teacher model 

将知识从一个"教师"神经网络（teacher ANN）迁移到一个"学生"神经网络（student ANN）。
最初由 Hinton 等人 于 2015 年提出。
学生模型被训练为尽可能接近教师模型的输出。
学生模型结构更简单、计算开销更低，同时在性能上尽量接近教师模型。


Details
▪ Instead of training the student model based on the hard labels only, it is in addition trained with the ==teacher model's output probability distributions (soft targets)==
▪ Soft targets encode much richer knowledge so that the student learns more nuanced decision boundaries.

学生模型不仅基于"硬标签（hard labels）"进行训练，
还会额外基于教师模型输出的**概率分布（soft targets）**进行训练。
👉 软标签（soft targets）包含更丰富的信息，使学生模型能够学习到更细致、更平滑的决策边界（decision boundaries）。


![](image/Pasted%20image%2020260219220719.png)

Hard label 只告诉学生：
"答案是谁"

Soft target 告诉学生：
"答案是谁 + 其他选项错到什么程度"


### 4.1.1 example

好，我们用一个**三分类例子**来直观理解 Knowledge Distillation 里的 **soft targets** 为什么更有信息量。

---


假设输入是一张图片，真实标签是：

> 🐱 Cat

只用硬标签（Hard Label）训练
真实标签的 one-hot 向量是：

```
Cat     Dog     Rabbit
1       0       0
```

学生模型只知道：

* Cat 是对的
* Dog 和 Rabbit 是错的
* 但不知道"错多少"

也就是说：

> Dog 和 Rabbit 在损失函数里被同样对待

但现实中，它们其实"错得不一样"。

---

使用教师模型的 Soft Targets
假设教师模型输出：

```
Cat     Dog     Rabbit
0.70    0.25    0.05
```

这表示：

* 教师认为是 Cat 的概率最高（70%）
* 但 Dog 也有一定相似度（25%）
* Rabbit 几乎不可能（5%）

👉 这背后隐藏的信息是：

> Cat 和 Dog 在特征空间里更相似
> Rabbit 和 Cat 差别更大

---

如果学生只用硬标签：

* 只学到"Cat 是 1，其它是 0"

如果学生用 soft targets：

* 学到 Cat > Dog > Rabbit
* 学到类别之间的"相似关系"
* 学到更平滑的决策边界

---


硬标签训练 → 决策边界更"陡峭"
soft target 训练 → 决策边界更"平滑"

这能带来：

* 更好的泛化能力
* 更稳定的训练
* 更接近教师模型的行为

---


Hard label 只告诉学生：

> "答案是谁"

Soft target 告诉学生：

> "答案是谁 + 其他选项错到什么程度"

这就是为什么蒸馏后的小模型可以：

* 参数更少
* 计算更少
* 但性能接近大模型



## 4.2 Variants

Multi-teacher Distillation
▪ Using more than one teacher model
▪ Student model learns from the averaged output of the teachers'
models
▪ Improves robustness and/or extends capabilities
使用多个教师模型（teacher models）
学生模型从多个教师模型的平均输出结果中学习
可以提高模型的鲁棒性，或扩展模型能力
👉 直观理解：
相当于"多个专家一起教学"，学生综合吸收不同模型的优势。


Adversarial Distillation
==Focusing on robustness:== Student model is trained to match the teacher's output. In addition, the student model is trained to match the teacher's output for tricky adversarial inputs. The goal for the student's model is to inherit the teacher model's robustness ==against input perturbations and attacks.==
Focusing on indistinguishability: Student model is trained to match the teacher's output. Discriminator learns to distinguish a student model's output from the teacher model's output, while the student learns to fool the discriminator. The goal of the student's model is to make its output indistinguishable from the teacher's output.
关注鲁棒性（Focusing on robustness）
学生模型被训练去匹配教师模型的输出
同时，还会针对**具有对抗性的棘手输入（adversarial inputs）**进行训练
目标是让学生模型继承教师模型在面对输入扰动或攻击时的鲁棒性

2️⃣ 关注不可区分性（Focusing on indistinguishability）
学生模型被训练去匹配教师模型的输出
同时引入一个判别器（discriminator）：
判别器学习区分学生模型输出和教师模型输出
学生模型学习去"欺骗"判别器
目标是让学生模型的输出在统计上无法与教师模型区分

Self-Distillation
▪ Same model (earlier version) acts as the teacher and student
▪ Reduce overconfidence and improve generalization
▪ Improve reasoning by refeeding the good chain of thought traces and results

同一个模型的"早期版本"同时充当教师模型（teacher）和学生模型（student）。

目标是：
降低模型的过度自信（overconfidence）
提高模型的泛化能力（generalization）
通过将优质的推理链（chain-of-thought traces）和结果重新输入模型进行训练，从而提升推理能力。



![[image/Pasted image 20251204110823.png]]


Key Takeaway
Even small models can "think" step-by-step if they are teached to copy the same structure of a big model's reasoning and not just the final answers
即使是小模型，也可以学会"逐步思考（step-by-step reasoning）"，
前提是训练它去模仿大模型的推理结构，而不仅仅是最终答案。

Method
1) Pick (complex) tasks and a teacher
2) Prompt the teacher model with the tasks by adding "Let's think step by step" (30 times for the same task, varying temperature). Result: a list of rationales and answer per prompt
3) Filter out generated samples that did not lead to the correct answer
4) Pick a student model and fine-tune it with the new generated corpus of training data

1️⃣ 选择（复杂）任务和一个教师模型

2️⃣ 给教师模型添加提示词：
"Let's think step by step"
对同一个任务生成 30 次结果（通过调整 temperature）。
结果：
得到一组"推理过程（rationales）+ 最终答案"的样本。

3️⃣ 过滤掉那些没有得到正确答案的样本
4️⃣ ==选择一个学生模型，用筛选后的新生成数据进行微调==


Evaluation
▪ Teacher model: GPT-3 (version: code-davinici-002, 175B) ▪ Student model: OPT-1.3B

Questions Investigated
▪ Does it result in higher-quality CoT in the student model? Yes ▪ Does the student model surpass the teacher model? No, but
close

学生模型是否能生成更高质量的 Chain-of-Thought（CoT）？
✅ 是的

学生模型是否能超越教师模型？
❌ 不能，但表现接近


## 4.3 SYMBOLIC CHAIN-OF-THOUGHT DISTILLATION



![[image/Pasted image 20251211101902.png]]


OpenAI only give 20 log probalities 


# 5 Quantization 

![](image/Pasted%20image%2020260219222044.png)
## 5.1 Intro

Concept
▪ Rescale weights and sometimes activations as well to get a more efficient model
▪ Smaller model has lesser memory and compute requirements
▪ Typical data types: FP32, FP16 and BF16
▪ Rescaled typically to: INT8 or INT4
▪ It is a tradeoff between model size and accuracy
==▪ Challenge: find best map for a high-precision data type to a low- precision data type while minimizing loss of accuracy==
▪ Technically, minimize information loss between two distributions 

* 对模型权重（有时也包括激活值）进行重新缩放（rescale），以获得更高效的模型
* 更小的模型意味着更低的内存占用和计算需求
* 常见的数据类型：FP32、FP16、BF16
* 通常会量化为：INT8 或 INT4
* 这是模型大小与精度之间的一种权衡（trade-off）
* 挑战在于：如何将高精度数据类型映射到低精度数据类型，同时尽量减少精度损失
* 从技术角度来看，本质是最小化两种分布之间的信息损失

---

Types of Quantization (When?)
▪ Post-training dynamic quantization (dynamic PTQ): activation quantized during inference
▪ Post-training static quantization (static PTQ): activation quantized based on a calibration dataset
▪ Quantization-aware training (QAT): quantization during pre- training

* **Post-training dynamic quantization（动态 PTQ）**
  在推理过程中对激活值进行量化
* **Post-training static quantization（静态 PTQ）**
  基于一个校准数据集（calibration dataset）对激活值进行量化
* **Quantization-aware training（QAT）**
  在预训练或训练阶段就引入量化机制

---

Types of Quantization (What?)
▪ Weight-only quantization
▪ Activation quantization (weights and activations) 
▪ KV-cache quantization


* **仅权重量化（Weight-only quantization）**
* **权重 + 激活值量化（Activation quantization）**
* **KV-cache 量化**（针对 Transformer 推理缓存优化）
---

Types of Quantization (How many target bits?)
▪ INT8 quantization 
▪ INT4 quantization
* **INT8 量化**
* **INT4 量化**


量化的核心就是：

> 用更少的 bit 表示模型参数
> 换取更小的模型体积和更快的推理速度
> 但会牺牲一定精度

如果你愿意，我可以帮你画一个"量化 vs 精度 vs 速度"的直观关系图，特别是针对 LLM 推理场景（比如 KV-cache 量化为什么很关键）。

## 5.2 

![](image/Pasted%20image%2020260219222608.png)

## 5.3 SYMMETRIC VS. ASYMMETRIC

Symmetric Quantization
▪ Requirement: ==Range of quantized integers is symmetric around zero==, e.g., INT8 : [-127,127]
▪ The scaling factor 𝑆 maps the largest absolute value 𝑣 of the input to the maximum representable integer
▪ Zero in the input maps exactly to zero in the quantized form
▪ Simpler and slightly faster, but less flexible for non-centered data
▪ Quantization error low for balanced input distribution
▪ Common for weight quantization

* 要求：量化后的整数范围必须以 0 为中心对称，例如 INT8：[-127, 127]
* 缩放因子 (S) 用于将输入中的最大绝对值 (v) 映射到可表示的最大整数值
* 输入中的 0 会精确映射为量化后的 0
* 实现更简单、速度略快，但对于非零中心分布的数据灵活性较差
* 当输入数据分布较为均衡（围绕 0 分布）时，量化误差较小
* ==常用于权重量化（weight quantization）==




Asymmetric (or Zero Point) Quantization
▪ Generalized form that allows any mapping between a real range and an integer range
▪ Slightly higher computational costs
▪ ==Quantization error is lower for asymmetric distributions==
▪ ==Common for activation quantization==

* 是一种更通用的形式，允许在任意实数范围和整数范围之间进行映射
* 计算开销略高
* 对于非对称分布的数据，量化误差更低
* 常用于激活值量化（activation quantization）


| 对称量化     | 非对称量化             |
| -------- | ----------------- |
| 以 0 为中心  | 可设置零点（zero point） |
| 计算更快     | 更灵活               |
| 适合权重     | 适合激活值             |
| 对偏移分布不友好 | 对偏移分布更友好          |



![](image/Pasted%20image%2020260219222707.png)


## 5.4 Methods 


Fixed-Point Quantization
▪ Commonly used to quantize float to integer
▪ Decimal point of float is fixed. One section of a target integer represents the integer part of the float and another section the fractional part, e.g., 4 bits represent the whole number part while 4 bits represent the fractional part.

INT8 Quants
▪ For static PTQ (needs calibration dataset) ▪ FP32 to INT8 quantization
▪ For weights and activations
▪ Quantization of activations

Blockwise Quants
▪ For static PTQ (needs calibration dataset)
▪ Quantize smaller parts (blocks) separately, each with its own scaling factor. All blocks apply symmetric quantization
▪ 2 variants: one scale per block or an additional scale bias for outliers
▪ Typically, 32 weights per block
▪ Concept of blocks is still used in popular methods today
▪ GGUF codes: Q4_0, Q4_1, Q8_0)with `Q<bits>_<variant>`

K-Quants
▪ For static PTQ (needs calibration dataset)
▪ Family of quantization schemes
▪ Introduces concept of superblocks: A superblock comprises smaller blocks. Each block gets a scaling factor. 1 additional extra scaling factor per superblock for outlier correction
▪ Introduces adaptive bit allocation: Important weights get more bits while less important weights get fewer bits.
▪ K-quants target weights only (only symmetric quantization)
▪ Dequantizes at inference on demand
▪ GGUF codes: Q4_K_M, Q4_K_S)with `Q<bits>_K_<variant>`
▪ Variants: small, medium or large quality tier
▪ Most popular quantization method for local LLM inference

**翻译如下：**

---

Fixed-Point Quantization（定点量化）

* 常用于将浮点数（float）量化为整数（integer）
* 将浮点数的小数点位置固定
* 目标整数的一部分 bit 表示整数部分，另一部分 bit 表示小数部分

  * 例如：8 bit 中

    * 4 bit 表示整数部分
    * 4 bit 表示小数部分

---

INT8 Quants（INT8 量化）

* 用于静态 PTQ（需要校准数据集）
* 将 FP32 量化为 INT8
* 可用于权重和激活值
* 也包括对激活值的量化

---

Blockwise Quants（分块量化）

* 用于静态 PTQ（需要校准数据集）
* 将权重划分为多个小块（blocks），分别量化
* 每个块有自己的缩放因子（scaling factor）
* 所有块采用对称量化

两种变体：

1. 每个块一个缩放因子
2. 每个块一个缩放因子 + 额外的缩放偏置用于处理异常值（outliers）

* 通常每个块包含 32 个权重
* "块"的概念至今仍广泛使用
* GGUF 编码格式：
  `Q<bits>_<variant>`
  例如：Q4_0、Q4_1、Q8_0

---

K-Quants

* 用于静态 PTQ（需要校准数据集）
* 是一类量化方案家族

引入的新概念：
🔹 Superblock（超级块）

* 一个 superblock 由多个小 block 组成
* 每个 block 有自己的缩放因子
* 每个 superblock 额外有一个缩放因子，用于修正异常值



自适应 bit 分配（Adaptive bit allocation）
* 重要权重使用更多 bit
* 不太重要的权重使用更少 bit

其他特点：

* 只针对权重进行量化（仅对称量化）
* 在推理时按需反量化（dequantize on demand）
* GGUF 编码格式：
  `Q<bits>_K_<variant>`
  例如：Q4_K_M、Q4_K_S
* 质量分级：small / medium / large
* 是目前本地 LLM 推理中最流行的量化方法

---

| 方法          | 核心特点          | 常见用途        |
| ----------- | ------------- | ----------- |
| Fixed-point | 固定小数点表示       | 基础整数映射      |
| INT8        | 标准 FP32→INT8  | 通用 PTQ      |
| Blockwise   | 分块独立缩放        | 经典 GGUF     |
| K-Quants    | 超级块 + 自适应 bit | 本地 LLM 主流方案 |



![[image/Pasted image 20251211102827.png]]


## 5.5 **GPT-GENERATED UNIFIED FORMAT (GGUF)**

Concept
▪ ==A binary format to store quantized LLMs==
▪ Developed by Georgi Gerganov (in context of llama.cpp)
▪ Permits to encode full-precision, partially quantized and fully quantized models
▪ Optimized for inference
▪ Unified standard across tools, OS and hardware
▪ Self-contained file including all information
▪ Versioning and backward compatibility

Format
▪ Metadata Layer: Model info & architecture, chat template ...
▪ Tokenization Layer: Vocabulary, special tokens, tokenizer type
▪ Quantization Layer: Quantization codes, number of bits used per weight, block size, scaling factors, ...
▪ Model Weights and Tensors: Embeddings matrix, attention weights, feed-forward layers, ...


* 一种用于存储量化 LLM 的二进制格式
* 由 **Georgi Gerganov**（在 llama.cpp 项目背景下）开发
* 支持编码：

  * 全精度模型
  * 部分量化模型
  * 完全量化模型
* 专为推理（inference）优化
* 作为跨工具、操作系统和硬件的统一标准
* 自包含文件（self-contained），包含运行模型所需的全部信息
* 支持版本管理和向后兼容

---

格式结构（Format）

元数据层（Metadata Layer）
* 模型信息与架构
* 聊天模板（chat template）
* 其他配置信息

分词层（Tokenization Layer）
* 词汇表（Vocabulary）
* 特殊 token
* 分词器类型（tokenizer type）

量化层（Quantization Layer）
* 量化编码（quantization codes）
* 每个权重使用的 bit 数
* block 大小
* 缩放因子（scaling factors）
* 等等

模型权重与张量（Model Weights and Tensors）
* 嵌入矩阵（Embeddings matrix）
* 注意力权重（Attention weights）
* 前馈网络层（Feed-forward layers）
* 等等

---

Filename Format
`<model>-<size>-<variant>-<version>.<quantization>.gguf`
model:
size:
variant: version: quantization:
model name
number of parameters (e.g., 7B) fine-tuned type (e.g., base, instruct, chat) version or release tag
quantization method used



![[image/Pasted image 20251211103203.png]]

Format: put the quantization res 
all scaling the 


# 6 Benchmarking 

## 6.1 Intro 
MLPerf

What is Benchmarking?
▪ An LLM benchmark is a standardized framework to assess the performance of an LLM
▪ It comprises sample data, a set of questions or tasks to test the LLM, metrics for evaluating performance and a scoring mechanism
▪ Benchmarking is the process of evaluating the performance of an LLM with an LLM benchmark
LLM 基准测试（benchmark）是一个标准化框架，用于评估大语言模型的性能。

它通常包括：
样本数据（sample data）
一组用于测试模型的问题或任务
评估性能的指标（metrics）
评分机制（scoring mechanism）
Benchmarking 是指使用这些基准测试来评估 LLM 性能的过程。

Why Benchmarking?
▪ Judge the performance of an LLM with respect to a given task ▪ Make LLMs comparable
评估 LLM 在特定任务上的表现
让不同 LLM 之间可以进行可比性对比

What are the challenges for Benchmarking?
▪ Nondeterminism of LLMs (benchmarking a second time might lead to a different score)
▪ Fast evolution of LLMs (benchmarks get too easy)
▪ LLMs become aware of being benchmarked
▪ Prompt sensitivity, contamination, etc.

LLM 的非确定性（Nondeterminism）
同一个模型再次测试，可能得到不同分数

LLM 的快速演进
模型能力提升很快，原有 benchmark 可能变得过于简单: 

模型对基准测试"知情"
- 模型可能在训练中见过 benchmark 数据（数据污染）
- 或专门针对 benchmark 优化
提示敏感性（Prompt sensitivity）与数据污染（Contamination）等问题

![](image/Pasted%20image%2020260219223613.png)


fast evolution of LLMs:    metrics hast range, when LLm evolves.  the range shrickn and reach 100% good proformance .   the task is too easy for the evolved LLM 


## 6.2 PROPERTIES OF INTEREST

Prompt sensitivity and contamination 

Capability
Knowledge recall, reasoning, code generation, language understanding, multilingual ability, long-context reasoning

Efficiency
Latency, memory footprint, load time, throughput, energy efficiency

Robustness
Input perturbation, noise robustness (misspellings), prompt sensitivity, quantization stability

Accuracy
Numerical accuracy, semantic fidelity, task consistency, dequantization correctness

Utility
Helpfulness, instruction following, conversation quality, code usability

Safety & Ethical
Toxicity (offensive, bias), hallucination, fairness, security, refusal

Generalization
Zero-shot generalization, few-show learning, transferability(domains), tool use

Comparative
Size vs. accuracy, speed vs. alignment, cost vs. performance, hardware portability

Reproducibility
Repeatability, cross-version consistency, prompt determinism, cross- judge stability

## 6.3 TERMS


Scenario
==A broad set of contexts/settings or a condition== und which an LLM's performance is being assessed respectively evaluated. Examples are question answering, reasoning, machine translation, text generation.

Task
More fine granular job the LLM is being evaluated to complete. A task comprises of subtasks. Task examples: Arithmetic, multiple choice. Subtask examples: Arithmetic Level 1, multiple choice for algebra

Metric
A qualitative measure to evaluate the performance of a an LLM on given scenarios and tasks.

Benchmark Dataset
A standardized collection of test sets used to evaluate LLMs on a given scenario or task.

Benchmark Metric
A qualitative measure used for a benchmark

Benchmark Scorer
A scoring function


## 6.4 Big Picture 


![[image/Pasted image 20251211105120.png]]



how cloud we jedget the result without reference 
- check the length of oiutputs, check formular , check the the meaning of context 

## 6.5 Metrics 

test the LLM,  or the ai system contains the LLM 

statisitcla reliabilty: fewer entry
low human subjectibly: human decide if it is in fact or not 


Most Important and Common Criteria
▪ Answer relevancy 
▪ Task completion 
▪ Correctness
▪ Hallucination
▪ Tool selection correctness 
▪ Contextual relevancy
▪ Responsible metrics
▪ Task-specific metrics
任务完成率（task completion rate）
响应时间（latency）
成本（cost per request）
稳定性（reliability）



LLM vs. System Evaluation
Metrics for LLM use cases are custom metrics for the specific task, independent of the system the LLM is embedded in. Hence, they are consistent across different implementations. Metrics for the LLM system target aspects of how the LLM system performs. Example: task completion.

面向 LLM 用例的指标（Metrics for LLM Use Cases）
是针对具体任务设计的自定义指标
与 LLM 所嵌入的系统无关
因此，在不同实现方式之间具有一致性
👉 例如： 如果任务是文本分类、问答或翻译，那么准确率、F1-score 等指标在不同系统中都是可比较的。


Properties of a Good Metric
▪ Quantitative: Computes a score
▪ Reliability: Behave consistently. Give same score for same output.
▪ Accurate: Truly represent performance.
▪ Validity: It measures what it claims to measure.
▪ Discriminative Power: The scores for different models should not concentrate on a specific region of the scale.
▪ Robustness: Resistance to prompt injection, adversarial prompts, misspellings, contamination, artificial distractors, etc.
▪ Bias-Free: No language bias, no cultural assumptions, no domain familiarity, etc.
▪ Scalability: Big context window, large sample sizes, many test items, etc.
▪ Statistical Reliability: Statistical significance
▪ Transparency, Interpretability: Reproducability
▪ Low Human Subjectivity: Multiple humans involved
▪ Adaptability, Longlivety: Cope with evolution of LLMs


----

## 6.6 REFERENCE-BASED METRICS

Classification Metrics
▪ Accuracy: How often did the LLM reason correct?
▪ Precision/Recall: Of all reasoning steps the LLM took, how many were correct? Of all reasoning steps that were needed to solve the task, how many did the LLM actually perform?
▪ F1-Score: Harmonic mean of precision and recall

Accuracy（准确率）：
LLM 有多少次推理是正确的？

Precision / Recall（精确率 / 召回率）：
精确率：在 LLM 执行的所有推理步骤中，有多少是正确的？
召回率：在完成任务所需要的所有正确推理步骤中，LLM 实际完成了多少？

F1-Score： 精确率和召回率的调和平均值（harmonic mean）。

![](image/Pasted%20image%2020260219230535.png)

F1 Source : both the precision and recall , thier weight is same  

$(1+beta^2)  multiple ((P x R) / (beta^2 mal P + R))$  
beta is wight more, give more focus to xx  . beata > 1 , give focus on recall


---

Ranking Metrics (used especially for RAG evaluation)
▪ Precision@K: Proportion of top-K items that are relevant
▪ Recall@K: Proportion of all relevant items retrieved within the top-K results.
▪ Normalized Discounted Cumulative Gain (nDCG@K): Measures ranking quality (higher weight to relevant items ranked near the top; the order plays a role)
▪ Hit Rate @K: Binary metric that checks if at least one relevant item appears in the top-K.
▪ Mean Reciprocal Rank (MRR@K): Average of the reciprocal ranks of the first relevant item for all queries in top-K.

Precision@K：
在前 K 个结果中，相关项所占的比例。

Recall@K：
所有相关项中，有多少出现在前 K 个结果里。

Normalized Discounted Cumulative Gain（nDCG@K）：
衡量排序质量的指标。
越靠前的相关项权重越高
排名顺序会影响评分

Hit Rate@K：
二值指标。
只检查在前 K 个结果中是否至少包含一个相关项。

Mean Reciprocal Rank（MRR@K）：
对每个查询，取第一个相关结果的排名倒数（1/rank），
再对所有查询取平均值。


![](image/Pasted%20image%2020260219230541.png)



ranking metrics

K items 
give many of list to LLM and give the k top relavant    


1 ist very good ranking
0 ist very bad ranking:   0 means there are 0 relevants tieims 

(1/T) x (sum 1/R)

T: number of test
R: ranking betwenn 1 and 0 

---

Deterministic Matching
▪ Exact Match
▪ Fuzzy Match: Allows for minor variations, such as ignoring whitespace or formatting.
▪ Word or Item Match: Verifies if the response includes specific fixed words or strings, regardless of full phrasing.
▪ JSON match: Matches key-value pairs in JSON output
▪ Unit test pass rate: Tracks whether generated code passes predefined test cases.



BLEU:  refenence and output    n-gram also  existed in reference .  结合 Panelity 使用 
ROUGHE-n:   regarding the order,   only word , not semenatic meaning conprision

![](image/Pasted%20image%2020260219230617.png)


Exact Match（精确匹配）
要求输出与标准答案完全一致。

Fuzzy Match（模糊匹配）
允许轻微差异，例如忽略空格、格式差异等。

Word or Item Match（关键词/项目匹配）
检查回答中是否包含特定的固定单词或字符串，而不要求完整句子完全一致。

JSON Match（JSON 匹配）
匹配 JSON 输出中的键值对（key-value pairs）。

Unit Test Pass Rate（单元测试通过率）
统计生成的代码是否通过预定义的测试用例。

---

Overlap-based Metrics
▪ BLEU (Bilingual Evaluation Understudy): Evaluates n-gram overlap (commonly up to 4). Focuses on precision; penalizes brevity. "How much of what the model wrote appears in the reference?"
▪ ROUGE-n (Recall-Oriented Understudy for Gisting Evaluation): Evaluates the specified n-gram overlap. Focuses on recall. "How much of the reference content did the model capture?"
▪ METEOR (Metric for Evaluation of Translation with Explicit Ordering).
![](image/Pasted%20image%2020260219230624.png)


BLEU（Bilingual Evaluation Understudy）
评估 n-gram 的重叠程度（通常最高到 4-gram）
侧重 精确率（precision）
会对过短的输出进行惩罚（brevity penalty）
核心问题是：
"模型生成的内容中，有多少出现在参考答案中？"

ROUGE-n（Recall-Oriented Understudy for Gisting Evaluation）
评估指定 n-gram 的重叠
侧重 召回率（recall）
核心问题是：
"参考答案中的内容，有多少被模型捕捉到了？"

METEOR（Metric for Evaluation of Translation with Explicit Ordering）
综合考虑：
精确率
召回率
词形变化（stemming）
同义词匹配
词序信息


## 6.7 REFERENCE-BASED AND REFERENCE-FREE METRICS

Semantic Similarity
▪ BERTScore: Using BERT embeddings and calculate cosine similarity. Scores are aggregated to provide precision, recall, F1.
▪ MoverScore: Based on BERTScore. Calculates Wasserstein distance to figure out how much effort is needed to transform text.
▪ Reference vs. Output: How similar is the reference to the output?
▪ Input vs. Output: How well does the output align with the input content?
▪ Context vs. Output: How well is the content provided by a RAG integrated into the output?

语义相似度（Semantic Similarity）
BERTScore：
使用 BERT 生成文本的嵌入表示（embeddings），并计算余弦相似度（cosine similarity）。
最终汇总为精确率（precision）、召回率（recall）和 F1 分数。

MoverScore：
基于 BERTScore。
通过计算 Wasserstein 距离（也称 Earth Mover's Distance），衡量将一个文本"变换"为另一个文本所需的最小代价。

-----

不同对比方式
Reference vs. Output：
参考答案与模型输出之间的语义相似度如何？

Input vs. Output：
模型输出与输入内容的对齐程度如何？

Context vs. Output（尤其适用于 RAG）：
模型输出是否正确整合了检索到的上下文信息？


![[image/Pasted image 20251211111949.png]]

----



![[image/Pasted image 20251211112320.png]]

How consistent: how often the response simliar if i give the same prompt 

the more similar they are, the more consistent is my LLM

让一个 LLM 来负责比较"参考答案"和"模型输出"。

非常灵活，因为它可以评估多种属性，例如：

句法结构（syntactic structure）

术语使用（terminology）

事实一致性（factual consistency）

风格（style）

语气（tone）

逻辑与因果关系（logical and causal relationships）

语义表达（meaning）

可读性（readability）等

也可以让 LLM 比较两个输出，判断哪个更合适。
评判标准可以写入系统提示（system prompt）中。

LLM-as-a-Judge 是目前最流行的评估方法之一。


SelfCheckGPT
生成多个回答并进行比较。
通过"自一致性检查（self-consistency check）"来检测幻觉（hallucinations）。

## 6.8 REFERENCE-FREE METRICS

![[image/Pasted image 20251211112524.png]]


Regular Expressions
▪ Keywords Containment ("Buzzword Bingo")
▪ Keywords Omissions
▪ Detect jailbreak attempts
▪ Detect repetitions, e.g., "as a language model,..." ▪ Structure checks

关键词包含检查（Keywords Containment / "Buzzword Bingo"）
检查输出中是否包含特定关键词。

关键词缺失检测（Keywords Omissions）
检测是否遗漏了必须出现的关键词。

检测越狱尝试（Detect jailbreak attempts）
识别提示注入或绕过安全限制的模式。
检测重复内容
例如重复出现类似
"as a language model, …"

结构检查（Structure checks）
检查输出是否符合预期格式，例如：
是否为 JSON
是否包含特定字段
是否包含标题、列表等结构


----


Text Stats
▪ ==Measure quantifiable text statistics==
▪ Examples: Text length, word count, sentence count
▪ Measure readability scores: What educational level is required to comprehend the sentence
▪ Language detection
▪ Filler word probability distribution, e.g., "the"
▪ Names entity count

衡量可量化的文本统计指标
示例包括：
文本长度
单词数量
句子数量
可读性评分（例如判断需要什么教育水平才能理解该文本）
语言检测（Language detection）
填充词概率分布（例如 "the" 等高频词）
命名实体数量（Named entity count）


----

![[image/Pasted image 20251211112741.png]]

Deterministic Validation
▪ Validating format and structure
▪ Examples: Syntactical correctness, required field contained (schema compliance), validate content (e.g., links still valid, non null), test code execution
▪ Measure readability scores 

验证输出的格式与结构

示例包括：
语法正确性（syntactical correctness）
是否包含必需字段（schema compliance）
内容校验（例如链接是否有效、字段是否为 null）
代码执行测试（test code execution）
可读性评分测量

👉 特点：
基于规则或明确标准，结果可重复、可自动化。


---


Model-based Scoring
▪ Using pre-trained ML models to score the outputs ▪ Classify text by language
▪ Evaluate sentiment or emotion
▪ Assess readability

文本语言分类
情感或情绪分析
可读性评估

## 6.9 MOST PROMINENT BENCHMARKS

![[image/Pasted image 20251211112848.png]]




MMLU & BBH (general reasoning)
Broader knowledge and "intelligence"

GSM8K, MATH, AIME and MATH-500 (math)
Numeric and symbolic reasoning

HumanEval & SWE Bench (coding)
Programming and debugging

WinoGrande, PIQA & CommonsenseQA (commonsense)
Everyday reasoning

FLORES-200 & XNLI (multilingual, translation)
Metrics for LLM use cases are custom metrics for the specific task, independent of the system the LLM is embedded in. Hence, they are

TrusthfulQA, RealToxicity & advBench (safety, truthfullness)
Factual and ethical behavior

LongBench & Needle-in-a-Haystack (long-context)
Context retention

HELM (academic), Open LLM Leaderboard (open model standard, MT-Bench/Chatbo Arena (real-world conversational quality
Metrics for LLM use cases are custom metrics for the specific task,


---

TRUTHFULQA

![[image/Pasted image 20251211112914.png]]


![[image/Pasted image 20251211113116.png]]

---
MMLU
![[image/Pasted image 20251211113313.png]]


---
SWE-BENCH
![[image/Pasted image 20251211113352.png]]



 how good LLM resolve the git issue 

---

HUGGING FACE: OPEN LLM LEADERBOARD
![[image/Pasted image 20251211113616.png]]


![[image/Pasted image 20251211113646.png]]

Prompt 
instction\_id\_list:   condition   to excute the test 
Kwargs: 

---

CUSTOM BENCHMARKING
![[image/Pasted image 20251211114039.png]]


compare the LLm's solution without the musterlosung in buch



