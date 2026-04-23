

# 1 AGENTIC LOOP


## 1.1 AI AGENT IN ACTION

![](image/Pasted%20image%2020260219005654.png)

![](image/Pasted%20image%2020260219005704.png)


![](image/Pasted%20image%2020260219005714.png)

![](image/Pasted%20image%2020260219005733.png)


![](image/Pasted%20image%2020260219005740.png)

![](image/Pasted%20image%2020260219005756.png)

![](image/Pasted%20image%2020260219005806.png)

![](image/Pasted%20image%2020260219005813.png)


## 1.2 HOW TO WORK WITH LLMS?

1 User Prompt 
LLM are stateless. Once they responded to a request, they forget the conversation. They are pretrained but the do not learn or keep state.

The history of the conversation must be provided in each LLM call so that the LLM keeps the context across calls.
![](image/Pasted%20image%2020260219005932.png)

----
2 system prompt 
![](image/Pasted%20image%2020260219010005.png)

A system prompt gives the LLM a persona: a roles it should play in the conversation.

---

3 Define tools the LLM can ask to be called for further support.

![](image/Pasted%20image%2020260219010038.png)



# 2 Prompt Format 

![](image/Pasted%20image%2020260219010151.png)



Concept
▪ An LLM is instruction fine-tuned using instructions in a particular LLM-specific format with LLM-specific special tokens
大语言模型（LLM）是使用特定于该模型的指令格式和特殊标记进行指令微调的。

▪ A prompt format is a formatting logic that specifies how an instruction fine-tuned LLM ==expects a structured prompt to be properly serialized.== It is the format the instruction fine-tuned LLM was conditioned on during fine-tuning.
▪ A ==chat template== is a specific type of a prompt format for multi- turn conversations. It is the format the chat fine-tuned LLM was conditioned on ==during fine-tuning for chat conversations==.
▪ Prompt format / chat template are dealt with by the model's tokenizer
▪ An LLM that is fine-tuned using a specific prompt format is well capable of handling structured inputs as long as the input is provided in the desired prompt format. In other words, inference results will be significantly lesser accurate if the prompt does not comply with the desired prompt format.
▪ Moreover, in case an instruction fine-tuned LLM, e.g., such as Llama 2, should be further fine-tuned, then training instructions in the LLM-specific prompt format will lead to more accurate results.
▪ Examples: ChatML (GPT-3.5/4, Qwen), Harmony (for OpenAI's open weight models), Alpaca, Mistral, Llama 3

Challenges
▪ No standard or gold chat template. There exist hundreds of different chat templates. The landscape of chat templates is very fragmented.
▪ No or only little documentation of the chat templates
▪ Still ongoing work and continues changes
▪ APIs for multiple LLMs lack behind


---


Optimized Towards Caching-efficiency
▪ ==Prompt result caching means that the model caches a result for a specific prompt for later reuse==
▪ Returning cached results are charged one-tenth of the price of a new generation by Anthrophic and OpenAI
▪ Returning a cached result for a prompt requires a prompt to have a prefix "similar" to a prompt that was used before within a caching time window
▪ A caching-efficient prompt format should put stable information that is not supposed to change during a dialogue at the front and non-stable information close to the last added suffix of the prompt
▪ Harmony defines the reasoning effort in the system message at the start. Changing the reasoning effort during the conversation would break the caching opportunity.


![](image/Pasted%20image%2020260219010723.png)



## 2.1 CHATML: A Chat template

Concept
▪ A chat template introduced by OpenAI in March 2023
▪ GPT-3.5 and GPT-4 have full chatML support
▪ The concept is adopted by many close and open-source projects (Qwen/Qwen2/Qwen2.5 have full support for ChatML)
▪ Input and output prompts are framed with special start (<|im_start|>) and end (<|im_end|>) tokens
▪ Each prompt has an associated role: system, user or assistant
▪ The system prompt defines the overall instructions for the LLM
▪ User corresponds to a human user or AI agent
▪ Assistant corresponds to the LLM
▪ Some model introduced the tool, tool_call, tool_resonse roles to represent a tool calls
▪ Claude took over the role-based conversation pattern
▪ Some Mistral/Mixtral models on hugging face were fine-tuned with
chatML

```
<|im_start|>system
You are travel agent and responsible to book holidays. Answer as concisely as possible.
<|im_end|>
<|im_start|>user
Book me a holiday.<|im_end|>
<|im_start|>assistant
Yes, of course. For what dates?<|im_end|> <|im_start|>user
For the summer break in 2026<|im_end|>
```


Limitations
▪ No instruction hierarchy. System prompts have same priority as untrusted prompts from user or AI agents. Prompt injection: user can override or negate system instructions. Guardrails need to be implemented outside of LLM.
▪ Supports only text. Community used conventions such as "Here is the image in base64:"
▪ No support for tool use. LLM has no notion of arguments, return types or execution boundaries.
▪ No separation between chain-of-thought and user-visible output. Hard to suppress invisibility of reasoning thoughts.
▪ No security model. Every section of the prompt is treated similar.
▪ Deprecated as of August 2023: OpenAI announced to no longer
document updates to ChatML

Status
▪ Still used by open-source community
▪ Open-source training datasets on Hugging Face ▪ Compatibility layers



## 2.2 CHAT TEMPLATES FOR LLAMA FAMILY

![](image/Pasted%20image%2020260219011045.png)



## 2.3 LACK OF A STANDARD

Every Model Provider Cooks its Own Soup
▪ Qwen uses ChatML + custom tags
▪ Before llama 4 models use <|begin_of_text|>,
<|start_header_id|>, <|end_header_id|>, <|eot_id|>
▪ Llama 4 models use <|header_start|>, <|header_end|>, <|eot|>
▪ Gemma models use <start_of_turn>, <end_of_turn>
▪ `Mistral use [INST], [/INST]`
▪ Phi model use <|system|>, <|assistant|>, <|user|>, <|end|>
▪ Kimi models use <|im_system|>, <|im_assistant|>, <|im_user|>, <|im_middle|>



Model-specific Sequence of Roles in Messages
▪ OpenAI allows arbitrary sequence of system/user/assistant
▪ Gemini requires at most one system message at the start
▪ Anthrophic requires exact one system message at the start, multiple consecutive messages with the same role are merged to one message.
▪ Qwen allows arbitrary role alternation
▪ Gemma 3 requires strict role alternation. System message is embedded into the first user message.
▪ SmolLM3 allows arbitrary system messages, but it considers only the first one.


Tangled Abstraction Levels
What to do if the chat history need to be summarized?
▪ Add summary to system prompt or first user message?
▪ What if more than one user? Should both user histories be merged?
▪ How does an agent encode examples for a task-solution request
(few-shot prompting)?



Jinja Templates
▪ ==Some open weights models contain the chat template as Jinja transformation instructions within the GGUF file==
▪ Jinja is a templating engine for Python. Jinja takes the chat template (in form of a Jinja transformation "template") and the content of the API call and transforms both into a single prompt that complies with the chat template format of the LLM
Jinja transformation instructions for Qwen1.5-110B-Chat
▪ ▪
De-facto standard for chat template transformations
At Hugging Face, there exists more than 100 Jinja templates


![](image/Pasted%20image%2020260219012539.png)


## 2.4 HARMONY   (a prompt format )


Concept
▪ Harmony was introduced by OpenAI in August 2025 
▪ GPT-oss models were trained on the harmony format 
▪ Only supported by OpenAI's Responses API

Roles
▪ System: identity of the model, knowledge cutoff, reasoning effort, available channels, build-in tools (python, browser)
▪ Developer: instructions and tool definitions (formerly denoted as "system prompt")
▪ User and assistant just like in chatML
▪ Tool: representing output of tool calls
▪ Instruction hierarchy*, in case of instruction conflicts: system (highest priority), developer, user, assistant, tool (lowest priority)

Channels
▪ To differentiate between user-facing responses and internal messages ▪ Channels: final (towards user), analysis, commentary

Responses
▪ ==Model's responses also comply to the chat template format==
▪ Responses contain the final output towards the user, but they also reveal "how the model reasons".
▪ The API might not reveal or is not capable to reveal any additional information given by the LLM through the response besides the result.

![](image/Pasted%20image%2020260219012737.png)

![](image/Pasted%20image%2020260219012621.png)



# 3 API 


![](image/Pasted%20image%2020260219110038.png)


Latest API version might not reflect latest prompt format


Stateful:  hide the reasoning (lead to the conclude) in the cache, do not provide the whole conversation history.   do not need provide the reasoning path in the stateful mode. 

Build-in tool:  active through api behind the scene 

## 3.1 CHAT COMPLETION API

给模型一段 prompt，模型补全（complete）后面的文本。
The Chat Completion API is an interface that allows developers to generate conversational responses from a large language model.
Uses HTTP requests to call the model
Accepts a structured messages array
Supports multi-turn conversations
Returns AI-generated text responses


![](image/Pasted%20image%2020260219110321.png)

Concept
▪ Introduced by OpenAI in June 2020
▪ REST API for base and instruction-tuned models
▪ GTP-3 model family
▪ Access (originally): Invite-only, Public from Nov. 2021 
▪ Declared legacy, Last update in July 2023

Technical Details
▪ End point: https://api.openai.com/v1/completions
▪ HTTP method: POST
▪ HTTP header with "content-type: application/json" and "Authorization: Bearer $OPENAI_API_KEY"
▪ No conversation history
▪ No streaming

---


Concept
▪ Introduced by OpenAI in March 2022
▪ De-facto standard for REST API for LLMs
▪ Adopted with modifications by Antrophic Claude (Messages API), Mistral (Chat Completion API)
▪ Varies slightly among LLMs Technical Details (of OpenAI)
▪ End point: https://api.openai.com/v1/chat/completions
▪ HTTP method: POST
▪ HTTP header with "content-type: application/json" and "Authorization: Bearer $OPENAI_API_KEY"
▪ Streaming (with server-sent events) or non-streaming (request/response)
Non-streaming 普通请求 → 等待模型生成完成 → 一次性返回
Streaming  一点点发送生成的 token
▪ Parameter support can differ depending on the model
▪ Older version: https://api.openai.com/v1/completions


![](image/Pasted%20image%2020260219110532.png)

Content of sample HTTP response
![](image/Pasted%20image%2020260219110613.png)



---

![](image/Pasted%20image%2020260219111124.png)

### 3.1.1 Streaming

▪ Server-sent events (SSE)
▪ Using long-lived TLS connection
▪ An SSE of the streaming Chat Completion API contains a delta object with the delta content of the response.
▪ Delta object can contain natural language text in the content attribute but also tool calls in the delta.tool_calls object.

![](image/Pasted%20image%2020260219111211.png)

### 3.1.2 Structured Output
▪ Enables JSON output
▪ Supported by GPT-4o and newer models
▪ For older models: JSON mode
▪ Set by the response_format attribute in the request
▪ If it is set to "type": "json_object", then JSON
mode is activated. JSON mode guarantees JSON-
formatted output.
▪ If it is set to "type": "json_schema" ,
"json_schema": {...}, then model will output JSON-formatted output complying to the provided
JSON schema.

![](image/Pasted%20image%2020260219111311.png)


### 3.1.3 Support for JSON Schema

ChatCompletionAPI of OpenAI supports only a subset of the JSON schema language

![](image/Pasted%20image%2020260219111335.png)


## 3.2 RESPONSES API



Concept
▪ Released in March 2025
▪ ==Stateful (optional): No need to provide the whole conversation history at each call==
▪ Build-in tool calls (e.g., file search, code interpreter, web search, image generation and MCP)
▪ Reponses API implements an agentic loop (Unifying Assistant API and Chat Completion API)
▪ Better performance for reasoning model, really? Why? 

Technical Details
▪ End point: https://api.openai.com/v1/responses
▪ HTTP method: POST
▪ HTTP header with "content-type: application/json" and "Authorization: Bearer $OPENAI_API_KEY"
▪ Streaming (with server-sent events) or non-streaming (request/response)
▪ Parameter support can differ depending on the model


![](image/Pasted%20image%2020260219111656.png)

![](image/Pasted%20image%2020260219111720.png)


---

旧 API 问题：
每次调用必须发送完整历史：
messages: [ ... 所有对话历史 ... ]

Responses API 可以：
服务器保存对话状态
你只发送新的 input



Implements Agentic Loop
它内部实现：
```
User input
   ↓
Model reasoning
   ↓
Decide tool
   ↓
Call tool
   ↓
Get tool result
   ↓
Continue reasoning
```

以前你必须自己写：
tool calling
tool execution
再次喂回模型

现在 API 内部帮你完成。

----


原生支持 multi-step reasoning

旧 API：
一次调用 = 一次生成
推理必须写在 prompt 里（chain-of-thought）

Responses API：
内部可以循环推理
可以在 reasoning 中间调用工具
可以分阶段思考

👉 更接近真正 Agent

## 3.3 LLM GATEWAY


1 Challenges
▪ ==Every LLM provider comes with its own prompt format and API: API fragmentation==
▪ Rates limits and cost models differ from model to model even within the same LLM provider
▪ Custom code for each LLM provider (vendor lock-in)
▪ Custom code for fallback solutions, load balancing among LLM providers
▪ Auditing for every LLM provider

每个 LLM provider：
不同 endpoint
不同 JSON 结构
不同参数命名
不同 tool call 格式


模式的；情态的；形式的

![](image/Pasted%20image%2020260219111905.png)

![](image/Pasted%20image%2020260219112011.png)


---

Solution
▪ An LLM gateway or LLM proxy (generally, a specific type of API gateway) in between the AI agent and multiple LLM providers
▪ Hides particularities of the LLM-specific APIs
▪ Unifies interface
▪ One-top-shop for policy enforcement such as rate limiting and auditing
▪ Load balancing
▪ Smart routing
▪ Guardrailing
▪ Fallback management

Technical Approach
▪ SDK (e.g., LangChain) vs. hosted solution ▪ Self-hosted vs. cloud-hosted
▪ On premises vs. third party operated
▪ Open source vs. proprietary solutions

每个 LLM Provider 都有：
不同 endpoint
不同 JSON 格式
不同 tool 结构
不同 streaming 格式

Gateway 负责：
👉 把统一格式 → 转换成 provider-specific 格式
👉 把 provider-specific 响应 → 转换回统一格式


---


Downsides
▪ Unification leads to simplification but at the expense of missing LLM-specific features
▪ Single point of failure
▪ Integration of new features of LLM-specific APIs lag behind
▪ Extra latency
▪ Additional fees charged by LLM gateway operator
▪ Data and Security exposure to LLM gateway operator
▪ No end-to-end security

▪ 接口统一带来简化，但会牺牲特定 LLM 的专有功能
（为了统一接口，可能无法使用各个 LLM 提供商的高级或特有功能。）

▪ 单点故障（Single Point of Failure）
（如果 Gateway 出现故障，所有 LLM 请求都会受到影响。）

▪ 对 LLM 特定 API 新功能的集成存在滞后
（LLM 提供商推出新功能后，Gateway 需要时间适配，无法立即使用。）

▪ 额外的延迟（Extra latency）
（多了一层转发与处理，会增加请求响应时间。）

▪ LLM Gateway 运营方可能收取额外费用
（除了 LLM 提供商费用，还可能需要支付 Gateway 服务费用。）

▪ 数据与安全暴露给 LLM Gateway 运营方
（请求与响应数据需要经过 Gateway，增加数据暴露风险。）

▪ 无法实现端到端安全（No end-to-end security）
（数据需要经过中间层，不再是应用与 LLM 提供商之间的直接加密通信。）




# 4 Context Engineering 

==Prompts are Model-specific==
▪ Anthropic recommends to structure prompts with XML tags
▪ OpenAI suggests to structure prompts with Markdown tags
▪ Gpt-3.5-turbo performs better if instruction is placed at start of prompt 
▪ Llama-2 performs better if instruction is placed at the end of a prompt ▪ Limited windows length
▪ Static prefix for caching efficiency

▪ Anthropic 建议使用 XML 标签来结构化 Prompt
`（例如使用 <instructions>、<context> 等标签来组织内容。）`

▪ OpenAI 建议使用 Markdown 标签来结构化 Prompt
（例如使用 ## 标题、列表、代码块等 Markdown 结构。）

▪ GPT-3.5-turbo 在将指令放在 Prompt 开头时表现更好
（模型对前缀内容更敏感，前置指令效果更稳定。）

▪ Llama-2 在将指令放在 Prompt 末尾时表现更好
（该模型对靠近结尾的指令更敏感。）

▪ 上下文窗口长度有限（Limited context window length）
（每个模型都有最大 token 限制，超过后可能截断或报错。）

▪ 使用静态前缀以提高缓存效率（Static prefix for caching efficiency）
（固定不变的前缀可以被系统缓存，从而降低计算成本和延迟。）


## 4.1 LLM SETTINGS


Temperature
Temperature as discussed in Large Language Model | Output | Sampling strategies
（数值越高，输出越随机；数值越低，输出越确定。）
Typical range = [0-1]


Top P
Recommendation: Only adjust temperature XOR Top P
Nucleus sampling as discussed in Large Language Model | Output | Sampling strategies
Typical range = [0,9-0,95]
Top P 使用 Nucleus Sampling（核采样） 方法：
只从累计概率达到 P 的 token 集合中采样
控制生成时的多样性
![](image/Pasted%20image%2020260219113314.png)



Max Length
Limit the number of output tokens


Stop Sequences
Specify text sequence that forces the LLM to stop outputting. For example, special tokens for tool calling so that an LLM stops immediately after an outputted tool call instruction for the AI agent.
指定一个文本序列，一旦模型生成该序列，就会强制停止输出。


Presence Penalty
Penalize words or tokens with a fixed penalty if they have appeared at least once in the conversation. Highly useful for AI agents because getting stuck in loops is a common challenge. For example, a specific tool should only be called once or only once with a specific parametrization.
Recommendation: only adjust presence XOR frequency penalty
如果某个词或 token 在对话中至少出现过一次，
就对它施加一个固定惩罚。

Frequency Penalty
Penalize words or tokens proportional to how many times it appeared in the conversation.
Typical range = [-2 (likely to repeat words), 2 (avoids to repeat words)]
Typical range = [0-1]
根据某个词在对话中出现的次数，按比例施加惩罚。
出现次数越多 → 惩罚越大 → 越不容易再次生成。

## 4.2 ZERO & FEW SHOT PROMPTING


Zero-shot Prompting
The prompt does not contain any examples or demonstrations. The LLM as assumed to understand the objective of the prompter. Works only with models that are trained with a vast amount of data*.
▪ Prompt 中不包含任何示例或演示。
▪ 假设 LLM 本身已经理解提问者的目标。
▪ 只有在经过大量数据训练的模型上才能有效工作。
模型训练数据足够大
模型已经学会通用任务模式
任务不太复杂

![](image/Pasted%20image%2020260219113416.png)


----

Few-shot Prompting
▪ The prompt contains examples or demonstrations.
▪ Examples and demonstrations can even improve performance of
LLMs that are capable to be prompted in a zero-shot manner.
▪ Can be interpreted as a lightweight, dynamic form of task-specific fine-tuning.
==▪ In-context learning: extrapolate examples to solve unseen but similar problems==
▪ Examples for AI agents: Show the model exemplarily how to reason or how to call tools
▪ Limitations: The more complex the to-be-learned pattern becomes the lower is the probability that the LLMs learns it.

▪ Prompt 中包含示例或演示（examples / demonstrations）。
▪ 即使模型可以 zero-shot 工作，示例仍然可以提升性能。
▪ 可以理解为一种轻量级、动态的任务特定微调。
▪ 属于 In-context Learning（上下文学习）：
模型通过示例推断模式，并解决类似但未见过的问题。
▪ 对 AI Agent 来说，可以示范：
如何推理
如何调用工具

▪ 局限性：
任务模式越复杂
LLM 成功学会该模式的概率越低

![](image/Pasted%20image%2020260219113456.png)

---


User
A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: We were traveling in Africa and we saw these very cute whatpus.
To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses
the word farduddle is:


## 4.3 PROMPT LENGTH

Longer Prompt, Better Result? It depends.
▪ Longer prompts do not necessarily lead to higher performance. They can even lead to decreased performance for reasoning about the given in-prompt context.*1

▪ The longer the context in the prompt, the lower might get the performance of the LLM, even when the retrieval of relevant information from the context works almost perfect.*
▪ The placement of the relevant information in the context is irrelevant to the correlation of context size and performance.



![](image/Pasted%20image%2020260219113546.png)

![](image/Pasted%20image%2020260219113601.png)


▪ 更长的 Prompt 并不一定会带来更高的性能。
在某些情况下，尤其是在需要基于 Prompt 内部上下文进行推理时，
过长的 Prompt 甚至可能会导致性能下降。*¹

▪ Prompt 中的上下文越长，LLM 的性能可能越低。
即使模型几乎能够完美地从上下文中检索到相关信息，
性能仍可能随着上下文长度的增加而下降。*²

▪ 相关信息在上下文中的位置并不会改变"上下文长度与性能之间的负相关关系"。
也就是说，无论重要信息放在前面、中间还是后面，
上下文越长，性能下降的趋势仍然存在。




## 4.4 PROMPT COMPRESSION

Prompt Compression
▪ Problem: The context window is limited. As latest research shows, a big context windows is not used uniformly by the LLM.
▪ Challenge: How to compress the context of a prompt to fit into the context window without loosing the key meanings?
▪ Motivation: Reduce costs (improve efficiency) and/or improve performance
▪ Most prompt compression methods fall into one of the following categories: RL-based methods, LLM scoring-based methods and LLM rewrite/annotation-based methods

大多数 Prompt 压缩方法通常属于以下几类：

基于强化学习的方法（RL-based methods）
基于 LLM 评分的方法（LLM scoring-based methods）
基于 LLM 重写/标注的方法（LLM rewrite/annotation-based methods）



![](image/Pasted%20image%2020260219122356.png)

---

Hard & Soft Prompts
▪ Hard prompts are formulated in natural language text. The compressor generates a different natural language text representing the original prompt.
▪ Soft prompts are partly or fully encoded as embeddings. The compressor g==enerates sequences of embeddings== representing the meaning of the prompt's context.

Exemplary Hard Prompt Methods
▪ SelectiveContext: identify and remove redundant or less informative parts by quantifying informativeness of lexical units using self-information)
▪ LLMLingua: similar to SelectiveContext but uses smaller LLM to judge about the informativeness of prompt sections
▪ Nano-Capsulator: summarizes (and as such keeps text fluent)
▪ LongLLM: like LLMLingua with longer context window
▪ LLMLingua-2: First, use LLM to label parts of a training text as important or unimportant (data distillation) to produce a training dataset. Then, train a word classifier to detect important words. A prompt is then first fed into the classifier and then reduced based on the classification per word.

▪ Hard Prompts（硬提示）
使用自然语言文本来编写 Prompt。
压缩器（compressor）生成的是另一段自然语言文本，用来表达原始 Prompt 的含义。

▪ Soft Prompts（软提示）
Prompt 的部分或全部内容以**向量嵌入（embeddings）**的形式编码。
压缩器生成的是一系列嵌入向量，用于表示 Prompt 上下文的语义。

![](image/Pasted%20image%2020260219122625.png)

## 4.5 CHAIN-OF-THOUGHT PROMPTING

cluster the question, get the reasoning path. get a representative example, give the example to the prompt 

Few-shot CoT Prompting
Give examples in the prompt on how to reason in order to come to the intended result.
Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, Wei et al., 2022
![](image/Pasted%20image%2020260219122957.png)

![](image/Pasted%20image%2020260219123234.png)

----

Zero-Shot CoT Prompting
Add "Let's think step by step." to the prompt to enforce step-by- step reasoning*.
Large Language Models are Zero-Shot Reasoners, Kojima et al., 2022
![](image/Pasted%20image%2020260219123022.png)

![](image/Pasted%20image%2020260219123249.png)

---


Automatic CoT （mix of two）
Add automatically-generated (using Zero-Shot CoT prompting) and diverse few-shot reasoning examples to the prompt.

![](image/Pasted%20image%2020260219123343.png)

![](image/Pasted%20image%2020260219123123.png)

## 4.6 SELF-CONSISTENCY

Self-consistency
▪ Situation: For a single task, there might exist different reasoning paths that lead to the same result. A wrong reasoning path taken by the LLM leads to a false result.
▪ Assumption: Correct reasoning paths are more likely to come to the same result than wrong reasoning paths.
▪ Solution: Step 1: Prompt with few-shot CoT examples, Step 2: Sample different reasoning paths (e.g., changing temperature, top-k sampling, nucleus sampling), 
Step 3: Select the result that is most consistent among sampled reasoning paths.


▪ **情境（Situation）：**
对于同一个任务，可能存在多种不同的推理路径（reasoning paths）能够得出相同的结果。
如果 LLM 选择了一条错误的推理路径，就会得到错误的结果。

▪ **假设（Assumption）：**
正确的推理路径更有可能得出相同的最终结果；
而错误的推理路径往往会得出不同的结果。

▪ **解决方案（Solution）：**

**步骤 1：**
使用 Few-shot Chain-of-Thought（少样本思维链）提示。

**步骤 2：**
采样多个不同的推理路径，例如通过调整：
* temperature
* top-k sampling
* nucleus sampling（top-p）

**步骤 3：**
在多个采样结果中，选择出现次数最多、最一致的最终结果。

![](image/Pasted%20image%2020260219124045.png)

## 4.7 TREE OF THOUGHTS


Tree of Thoughts (ToT)
▪ Similar to Self-consistency, ==but constructing a tree of thoughts instead of parallel reasoning paths==
▪ Breadth-first and depth-first search applied to efficiently find a promising reasoning path
▪ ▪
With look ahead to predict what might happen if continue down a path, and backtrack to go back if a path leads to a dead end or false outcome (only possible with a certain class of problems)
Advantage of self-consistency: evaluation of each step before continuing, sample-efficient for problems with large solution space

▪ 类似于 Self-consistency（自一致性）方法，
但不是生成多个并行的推理路径，
而是构建一个"思维树"（tree of thoughts）。

▪ 使用广度优先搜索（Breadth-First Search, BFS）和深度优先搜索（Depth-First Search, DFS），
以高效地寻找有前景的推理路径。

▪ 通过前瞻（look-ahead）机制预测如果继续沿当前路径推理会发生什么，
如果某条路径通向死胡同或错误结果，则可以回溯（backtrack）。
（这种机制只适用于某些特定类别的问题。）

▪ 相比 Self-consistency 的优势：
在继续下一步之前，可以对每一步进行评估；
对于解空间较大的问题来说，更加节省采样成本（sample-efficient）。

![](image/Pasted%20image%2020260219124055.png)

![](image/Pasted%20image%2020260219124102.png)

## 4.8 ToT 和 Self-consistency

题目
你有数字：
2, 3, 4

目标：
用 + − × ÷ 得到 14

---

### 4.8.1 Self-consistency
生成多个不同推理路径
每条路径独立完成\
最后投票选出现最多的结果

Self-consistency
不检查中间步骤
不回头修改错误路径
只是"多试几次"

![](image/Pasted%20image%2020260219153848.png)
![](image/Pasted%20image%2020260219153859.png)

---
### 4.8.2 ToT
构建"推理树"，逐步扩展，每一步评估是否有希望。

![](image/Pasted%20image%2020260219124611.png)

![](image/Pasted%20image%2020260219124624.png)

| 特点       | Self-consistency | Tree of Thoughts |
| -------- | ---------------- | ---------------- |
| 结构       | 多条独立路径           | 树状结构             |
| 是否回溯     | ❌                | ✅                |
| 是否评估中间步骤 | ❌                | ✅                |
| 适合问题     | 数学/推理            | 复杂规划             |
| 计算方式     | 多次采样             | 搜索算法             |



## 4.9 META PROMPTING

Meta Prompting
▪== Focus on structural and syntactical aspects of tasks and problems rather than content-specific examples.==
▪ "A Meta Prompt is an example-agnostic structured prompt designed to capture the reasoning structure of a specific category of tasks [...] it outlines the general approach to the problem"
▪ For each category of tasks there exists a structured meta prompt that described how to solve the problem rather than giving few- shot examples.
▪ Prompts for complex task and problems can be decomposed into meta prompts for each subtask
▪ Advantage: token-efficient (no list of few-shot examples required), meta prompts are more generalized which means improved applicability to similar problems
▪ Disadvantage: manual creation of meta prompts


▪ 关注任务或问题的结构和语法层面，而不是具体内容示例。
（强调"如何解决问题"的结构，而不是提供具体案例。）
▪ "Meta Prompt 是一种与具体示例无关的结构化提示，旨在捕捉某一类任务的推理结构……它概述了解决问题的一般方法。"
▪ 对于每一类任务，都可以设计一个结构化的 Meta Prompt，
用来描述如何解决该类问题，而不是提供 few-shot 示例。
▪ 对于复杂任务或问题，可以将 Prompt 分解为多个子任务，
并为每个子任务设计相应的 Meta Prompt。

优点（Advantage）
▪ Token 效率高（token-efficient）
不需要列出大量 few-shot 示例。

▪ 更具泛化能力（more generalized）
由于 Meta Prompt 抽象的是问题的结构，因此更适用于类似问题。

缺点（Disadvantage）
▪ 需要人工设计 Meta Prompt
（创建高质量的 Meta Prompt 需要经验和人工设计。）


---

![](image/Pasted%20image%2020260219153303.png)

![](image/Pasted%20image%2020260219153315.png)

| Few-shot | Meta Prompt |
| -------- | ----------- |
| 给示例      | 给解题结构       |
| 模仿答案     | 遵循步骤        |
| 内容导向     | 结构导向        |
| token 多  | token 少     |
| 泛化弱      | 泛化强         |

![](image/Pasted%20image%2020260219153334.png)

## 4.10 RETRIEVE THEN SOLVE PROMPTING： 通过 Prompt 先让模型从上下文中检索出与问题相关的信息。


▪ Core idea: Separate retrieval from reasoning
▪ First, prompt to retrieve information of the context that is relevant
to answer the question
▪ Second, create a new prompt that contains the question and only the relevant information of the original context

▪ 核心思想（Core idea）：
将"信息检索"和"推理过程"分开进行。

▪ 第一步：
通过 Prompt 先让模型从上下文中检索出与问题相关的信息。

▪ 第二步：
构造一个新的 Prompt，其中只包含：

原始问题
与问题相关的上下文信息
然后再让模型进行推理并给出答案。

因为：
上下文可能很长
很多内容无关
过长的 Prompt 会降低性能
推理阶段不需要所有信息

所以：
先"过滤信息"，再"做推理"。

| 普通做法       | Retrieve-then-Solve |
| ---------- | ------------------- |
| 全部上下文 + 问题 | 先筛选，再推理             |
| 上下文很长      | 上下文精简               |
| 容易分心       | 更聚焦                 |
| 成本高        | 成本低                 |



![](image/Pasted%20image%2020260219153442.png)

![](image/Pasted%20image%2020260219153457.png)
# 5 Model Context Protocol

## 5.1 Intro 

𝑁𝑥𝑀 problem
where 𝑁 is the number of AI Agent implementations/frameworks and 𝑀 the number of tools
![](image/Pasted%20image%2020260219154023.png)

The model context protocol (MCP) standardizes the way an AI application such as an LLM-based ==AI agent interacts with resources & tools==.
用于标准化 AI 应用（例如基于 LLM 的 AI Agent）与资源和工具之间的交互方式。

![](image/Pasted%20image%2020260219154033.png)

MCP 是一个标准协议，用来规范 LLM Agent 如何安全地访问工具和数据。

它解决的问题是：
不同工具接口不统一
Agent 访问数据缺乏边界控制
缺乏标准的人类交互机制
不同系统之间难以协作


User
   ↓
MCP Client  ←（控制权在这里）
   ↓
MCP Server  ←（提供资源和工具）
   ↓
Files / Database / APIs / Tools

-----

ARCHITECTURE

![](image/Pasted%20image%2020260219154144.png)


---------

PROTOCOL STACK

![](image/Pasted%20image%2020260219154311.png)


---

## 5.2 OFFERED FEATURES (CAPABILITIES) of MCP client and server 

MCP client
Roots
Client-controlled scope boundaries that specify which parts of the client's context or data the server is allowed to see and operate on

由客户端控制的作用域边界，
用于指定服务器可以查看和操作的客户端上下文或数据范围。
（即：客户端决定服务器能访问哪些数据。）



Elicitation
==Allows a server to request structured input or confirmation from a human user via the clien==t, enabling explicit human-in-the-loop interaction

允许服务器通过客户端向人类用户请求结构化输入或确认，
从而实现明确的"人类在环（Human-in-the-loop）"交互。
（例如：服务器请求用户确认某个操作。）


Sampling
Allows a server to request that the client generate language-model output (text) using its local or configured LLM under the client's control

允许服务器请求客户端在其本地或已配置的 LLM 上生成语言模型输出（文本），
并且该生成过程由客户端控制。
（即：服务器可以请求客户端调用本地 LLM 生成文本。）

---

MCP Server
Prompts
Reusable interaction templates that guide the client and its LLM through structured tasks or workflows
可复用的交互模板，
用于引导客户端及其 LLM 完成结构化任务或工作流程。
（类似预定义的任务执行模板。）

Resources
Addressable pieces of data or context (such as files, documents, or datasets) that clients can discover and consume within defined roots.
可寻址的数据或上下文内容
（例如文件、文档或数据集），
客户端可以在定义好的 Roots 范围内发现并使用这些资源。


Tools
Callable operations that a client may invoke to perform specific, well- scoped actions as part of an LLM-driven or agentic workflow

客户端可以调用的操作，
用于在 LLM 驱动或 Agent 工作流程中执行特定且范围明确的动作。

（例如：读取文件、查询数据库、发送请求等。）



## 5.3 例子来理解 **MCP Client** 和 **MCP Server** 的角色

好的，我们用一个非常具体和现实的例子来理解 **MCP Client** 和 **MCP Server** 的角色。

AI 编程助手
假设你正在使用一个类似 **Cursor** 或 **Claude Desktop** 的 AI 编程/写作工具，这个工具集成了 MCP 协议。

---

MCP Client（客户端）
**它是什么**：就是你直接与之交互的那个**应用程序本身**。
*   **具体例子**：`Cursor IDE` 或 `Claude Desktop 应用`。
*   **核心职责**：
    1.  **提供用户界面**：你输入问题、查看代码、和 AI 对话的窗口。
    2.  **托管/调用核心 LLM**：它内置或连接了像 Claude-3.5-Sonnet、GPT-4 这样的核心大语言模型。
    3.  **管理"根路径"和权限**：它知道允许 MCP Server 访问你电脑上的哪些文件夹（比如 `~/projects/my_app`，但排除 `~/Documents/personal`）。
    4.  **协调工作流**：接收你的指令，决定何时调用哪个 MCP Server 的工具，并将结果整合后呈现给你。

---

MCP Server（服务器）
**它是什么**：一个独立的、提供**特定功能扩展**的后台服务。它不是一个完整的应用，而更像一个"插件引擎"。
*   **具体例子**：
    *   **`mcp-server-filesystem`**：一个提供文件系统读写权限的服务器。
    *   **`mcp-server-git`**：一个提供 Git 操作（如提交、查看历史、创建分支）的服务器。
    *   **`mcp-server-sqlite`**：一个允许你查询本地 SQLite 数据库的服务器。
    *   **`mcp-server-weather`**：一个可以查询天气 API 的服务器。

---

一次完整的交互流程示例

**你的指令**（在 Cursor 中对 AI 说）：
> "帮我在当前项目中创建一个新的组件 `Button.jsx`，然后提交到 Git，并附上提交信息 'feat: add Button component'。"

**幕后流程**：

1.  **MCP Client（Cursor）** 收到你的自然语言指令。
2.  **Client 的核心 LLM** 分析指令，将其分解为需要调用的"工具"：
    *   第一步：需要**写文件** → 调用 `filesystem` 服务器的 `write_file` 工具。
    *   第二步：需要**执行 Git 命令** → 调用 `git` 服务器的 `git_commit` 工具。
3.  **Client 调用 MCP Server**：
    *   Client 向 `mcp-server-filesystem` 发送一个结构化请求："请在路径 `/projects/my_app/src/components/Button.jsx` 创建文件，内容为 `...`。"
    *   `filesystem` 服务器执行操作，返回成功或失败信息。
    *   然后，Client 再向 `mcp-server-git` 发送请求："请在路径 `/projects/my_app` 执行 `git add` 和 `git commit -m 'feat: add Button component'`。"
    *   `git` 服务器执行操作，返回提交结果。
4.  **Client 整合与回复**：
    *   Client 收到所有工具的执行结果后，用核心 LLM 组织一段自然的语言回复你：
    > "已成功创建 `Button.jsx` 文件并提交到 Git，提交哈希是 `a1b2c3d`。"

---

### 5.3.1 关键类比与总结

| 概念 | 类比 | 现实例子 |
| :--- | :--- | :--- |
| **MCP Client** | **"主控台"或"指挥中心"** | **Cursor IDE**, **Claude Desktop**, 未来可能整合的 **VS Code**, **Obsidian** 等。 |
| **MCP Server** | **"专用工具箱"或"外挂模块"** | **文件工具包**、**Git工具包**、**数据库连接器**、**Jira 查询器**、**天气插件**等。 |
| **MCP 协议本身** | **"工具箱的连接标准"** | 就像 USB-C 接口，让任何符合标准的"工具箱"（Server）都能插到"主控台"（Client）上使用。 |

**核心价值**：
*   **安全可控**：Client 控制权限（Roots）。Git 服务器只能操作 Git，不能乱删你的文件。
*   **模块化与生态**：开发者可以轻松创建新的 Server（如连接公司内部 API 的 Server），任何支持 MCP 的 Client 都能立即使用它，无需等待 Client 自身更新。
*   **体验统一**：你可以在不同的 Client（如 Cursor 或 Claude Desktop）中使用同一套熟悉的工具（Servers），获得一致的 AI 增强体验。





## 5.4 Lifecycle 
MCP is a stateful protocol. The lifecycle needs to be managed.
![](image/Pasted%20image%2020260219155007.png)

----
initialization Phase 

![](image/Pasted%20image%2020260205110528.png)


![](image/Pasted%20image%2020260205110624.png)

![](image/Pasted%20image%2020260219155244.png)


----
GET TOOL LIST FROM MCP SERVER

![](image/Pasted%20image%2020260219155302.png)

![](image/Pasted%20image%2020260219155317.png)


----

CALL TOOL

![](image/Pasted%20image%2020260219155327.png)

![](image/Pasted%20image%2020260219155332.png)