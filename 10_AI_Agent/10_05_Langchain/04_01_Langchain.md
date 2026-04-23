
langChain
langGraph:  give you the trace of whole flow of information/call 


![](image/Pasted%20image%2020260115102020.png)

# 1 MODEL LAYER


## 1.1 MINIMUM EXAMPLE

![](image/Pasted%20image%2020260219161208.png)

Timeout for calling the model
Maximum number of retries in case of a failure
Model Name
Maximum tokens of the generated response
The model is called with a user message

Requirement
Environment variable OPENAI_API_KEY need to contain
the API key provided by OpenAI



## 1.2 MINIMUM RESPONSE 

Text response of the model

LangChains response object. This is not the response object provided by calling OpenAIs API. LangChains abstracts away the particularities of each model providers response objects

![](image/Pasted%20image%2020260219161314.png)


## 1.3 ACCESSING A LOCALLY HOSTED MODEL

Tip
Make use of LMStudio to download appropriate open-source models and run them locally. LMStudio allows to open a ChatCompletion- compliant local REST end point.

Creates a model reference that refers to an API that complies to the OpenAIs ChatCompletion API pseudo-standard.

Pointing to the ChatCompletion- compliant end point as provided by the locally-hosted model inference server.

For locally hosted models there is no API key required unless the model inference server is set up to ask for credentials of any kind, e.g., API key

使用 LMStudio 下载合适的开源模型并在本地运行。LMStudio 允许你开启一个兼容 ChatCompletion 接口的本地 REST 端点。

创建一个模型引用，该引用指向一个符合 OpenAI ChatCompletion API 伪标准的 API。

将模型指向由本地部署的模型推理服务器提供的、兼容 ChatCompletion 的端点。

对于本地托管的模型，不需要 API Key，除非该模型推理服务器被配置为需要某种凭证，例如 API Key。


![](image/Pasted%20image%2020260219161936.png)


## 1.4 ACCESSING A SELF-MANAGED BUT REMOTELY HOSTED MODEL


Setup
▪ Inference engine: Ollama
▪ Inference location: Remote
▪ Loaded model: llama3.3:70b
▪ LangChain's API abstraction: ChatOllama
▪ API access type: Authentication bearer token



Creates a model reference that refers to an API of the Ollama inference engine.

Pointing to Ollama-compliant end point as provided by the remotely-hosted model inference server Ollama.

LangChain's model API abstractions natively support only API key–based authentication. A bearer token, instead, needs to be passed explicitly as an HTTP parameter.


创建一个模型引用，该引用指向 Ollama 推理引擎 的 API。

将模型指向由远程托管的 Ollama 模型推理服务器提供的、兼容 Ollama 的接口端点。

LangChain 的模型 API 抽象层原生只支持基于 API Key 的身份认证。如果使用 Bearer Token，则需要通过 HTTP 参数显式地传递该令牌。




## 1.5 MODEL CALLS ARE STATELESS


![](image/Pasted%20image%2020260219162245.png)


## 1.6 MODEL CALLS WITH CONVERSATION HISTORY

On the model layer, short-term memory (conversation history) has to be handled manually with LangChain.

![](image/Pasted%20image%2020260219162314.png)


## 1.7 MESSAGE OBJECTS

Use LangChain message objects instead of a string-based approach for safety and extensibility reasons. Prevents
typos in roles and enables validation at compile time or by the IDE.

![](image/Pasted%20image%2020260219162415.png)

## 1.8 BATCHING OF MODEL CALLS

Multiple conversations can be batched into a single model call. If supported by the inference engine, then both calls can be executed in parallel.

The response object is a list of responses: a response object for each batched model call.

![](image/Pasted%20image%2020260219162440.png)

![](image/Pasted%20image%2020260219162933.png)


---

BATCHING OF MODEL CALLS WITH LIMITED PARALLELISM

Specify the maximum number of parallel model calls send to the model by LangChain. Once the model responses
with the result for the first two models calls, LangChain will submit the last model call.

![](image/Pasted%20image%2020260219162928.png)


## 1.9 MODEL'S LACK OF KNOWLEDGE

Watch out: Some of the latest OpenAI online models have access to tools such as time and calculators; consequently, they act as AI agents. In this case the response might already contain the desired answer.

![](image/Pasted%20image%2020260219163045.png)
Response: I can't determine today's sunset time without the current date and a specific location (city/coordinates). Please provide both, and I'll calculate it...



----

![](image/Pasted%20image%2020260219163122.png)

A model can't call a tool. It can just ask the agentic control logic to call the tool on its behalf.

该装饰器向 LangChain 框架表明，后续定义的函数应被解释为一个工具（tool）。
函数的描述（description）可以帮助模型判断何时应当调用该函数作为工具。或者，也可以在函数的 docstring 中提供该描述。

函数定义部分。参数列表中的类型提示（type hints）是必需的。

将该函数作为工具注册给模型。这个步骤本身不会触发模型调用。只有在模型被真正调用时，工具的定义才会一并传递给模型。

返回结果中不会包含任何生成的文本，而是包含一个工具调用请求（tool call request），其中的参数值是从用户提示中提取出来的。


The decorator signalizes the LangChain framework that the following function should be interpreted as a tool. The description helps a model to decide when to use the function as a tool. Alternatively, the docstring of the function can also contain the description.

Function definition. Type hints in parameter list are mandatory.

Introduces the function as a tool to the model. This does not lead to a model call. Tool definitions are provided once a model gets invoked.

The result doesn't contain any generated text but a tool call request with the parameters values extracted from the user's prompt:



## 1.10 IMPLEMENTATION OF A TOOL CALL (AGENTIC LOOP)


Adding the AI generated response to the conversation history.
Iterates over all potential tools call requests in the AI- generated response.
If the provided tool should be called according to the model, then call the function not directly but with invoke. Otherwise, the function arguments are not validated by LangChain and the invocation becomes invisible to LangChain, which might cause problems in downstream tasks in case the model calls are chained.
Package the function call's result into and tool message and add it to the conversation history.
Result: Sunset in Berlin today is at 16:30 local time (4:30 PM).

将 AI 生成的回复添加到对话历史中。

遍历 AI 生成回复中所有可能的工具调用请求（tool call requests）。

如果根据模型的判断需要调用某个工具，则不要直接调用该函数，而应通过 invoke 方法调用。否则，函数参数将不会被 LangChain 校验，并且该调用对 LangChain 来说是不可见的。如果模型调用是链式执行的，这可能会在后续任务中引发问题。

将函数调用的结果封装为一个工具消息（tool message），并将其添加到对话历史中。

结果：今天柏林的日落时间是当地时间 16:30（下午 4:30）。



![](image/Pasted%20image%2020260115102850.png)

![](image/Pasted%20image%2020260219163659.png)

## 1.11 Structured output 

STRUCTURED OUTPUT

![](image/Pasted%20image%2020260219164108.png)

![](image/Pasted%20image%2020260219164153.png)


Specify the structure of the desired output using a JSON schema. Each model provider supports a different subset of all possible JSON schema keys.

Introduce the JSON schema to the model. This does not lead to a model call. Desired output structures are handed over to the model during invocation.


![](image/Pasted%20image%2020260115104225.png)


---

Beispiel tool 

![](image/Pasted%20image%2020260115104458.png)

![](image/Pasted%20image%2020260115105312.png)

agent invoke 的时候 ， 给如自己的 定义的 tool (就是 python function ), 不用给如 tool 的 input value,   
agent  根据  prompt message 自己决定 什么时候 调用 那些 tool , 
在 output 中 看 那些 tool  被调用了

----

![](image/Pasted%20image%2020260115110829.png)


![](image/Pasted%20image%2020260115110902.png)


== those fucntion   was call every time , every loop
before_model  function
after_model function


before_agent  function
after_agent  function

---

![](image/Pasted%20image%2020260115111058.png)



warp_model_call 
warp_tool_call   
每个 tool/model   被call 的时候  warp_xxx_call 中定义的 function 会之后 被连带的called 

![](image/Pasted%20image%2020260115111013.png)


![](image/Pasted%20image%2020260115111218.png)


----



# 2 Agent Layer 


## 2.1 AGENT WITHOUT A TOOL

![](image/Pasted%20image%2020260219164415.png)

Creates an agent and specifies the model to be used by the agent. Optionally, a system prompt for the model can be provided.
Executes the agent

The result is a LangChain object. It contains besides other the conversation history:
HumanMessage(content='What is a tree?' [...] AIMessage(content='"Tree" has a few different, but related, meanings. Here are the main ones:\n\n- Biological tree (the plant)\n - A perennial woody plant with a single main trunk (often) that grows tall and has branches with leaves. It has secondary growth [...]


## 2.2 AGENT WITH A TOOL

![](image/Pasted%20image%2020260219164524.png)


## 2.3 TOOL DEFINTION WITH JSON SCHEMA OF PARAMETERS

JSON schema of the tool's parameters. The description of each attribute helps the model to fill
in the correct values and to output the arguments in a correct format within a tool call request.
Specifies the JSON schema to serve as the specification of the parameters of the tool.

![](image/Pasted%20image%2020260219164548.png)


工具参数的 JSON Schema。
每个属性（attribute）的描述可以帮助模型填写正确的值，并在工具调用请求中以正确的格式输出参数。

该 JSON Schema 用作工具参数的规范说明（specification）。


## 2.4 AGENT WITH MULTIPLE TOOLS

![](image/Pasted%20image%2020260219165403.png)


## 2.5 AGENT NEEDS TO CALL TOOLS IN A SPECIFIC ORDER

The model reasons about when to call what tool based on the conversation history.

![](image/Pasted%20image%2020260219165501.png)


![](image/Pasted%20image%2020260219165533.png)

## 2.6 MAKING THE AGENT TRANSPARENT

Are Agents Black Boxes?
An agents is invoked, executes and provides a result. However, the execution may involve multiple agentic loops.
LangChain's provides options to "look into" agents during execution and to run custom code at specific points in the agent's execution flow:
Core agentic loop
- Handlers (for callbacks): Follows observer pattern and do not own the execution of, e.g., the model. Used primarily to log, trace and debug.
- Middleware (node-style and wrap-style hooks): wrap-style hooks are execution wrappers around the model or tool calls. They control the execution. They allow to execute logic before and after the model call and they can even shortcut the model call if the model is not needed, e.g., because of existing cached results from former model calls with same prompt. Node-style hooks are adapters that participate in execution. They can't stop the execution flow but are allowed to change prompts, models, tools, even the conversation history, etc..

一个 Agent 被调用后会执行任务并返回结果。
然而，这个执行过程可能包含多个 agentic 循环（agentic loops）。

LangChain 提供了多种方式，可以在 Agent 执行过程中"查看内部情况"，并在执行流程的特定节点运行自定义代码。

---

核心 Agent 循环（Core agentic loop）

1️⃣ Handlers（回调处理器）
* 遵循观察者模式（observer pattern）。
* 不拥有模型等组件的执行控制权。
* 主要用于日志记录（logging）、追踪（tracing）和调试（debugging）。

---

2️⃣ Middleware（中间件）

分为两种类型：

 🔹 Wrap-style hooks（包裹式钩子）
* 是围绕模型或工具调用的执行包装器（execution wrappers）。
* **拥有执行控制权**。
* 可以在模型调用之前和之后执行自定义逻辑。
* 甚至可以跳过模型调用（shortcut），例如：

  * 当检测到相同 prompt 的缓存结果已经存在时，可以直接返回缓存结果，而不再调用模型。

🔹 Node-style hooks（节点式钩子）
* 作为执行流程中的适配器（adapter）参与执行。
* **不能终止执行流程**。
* 但可以修改：
  * prompt
  * 模型
  * 工具
  * 对话历史等内容

---

Agent 并不是完全的黑盒。
LangChain 提供了：

* 回调（Handlers）用于观察
* 中间件（Middleware）用于干预和控制执行流程

因此，开发者可以对 Agent 的执行过程进行监控、修改，甚至优化执行路径。


![](image/Pasted%20image%2020260129102053.png)

---


how to bypass the model call
like you have already asked the same question. then next time with the same example, we can pass the modle call 

![](image/Pasted%20image%2020260129103316.png)


### 2.6.1 HANDLER

Handlers（回调处理器）
* 遵循观察者模式（observer pattern）。
* 不拥有模型等组件的执行控制权。
* 主要用于日志记录（logging）、追踪（tracing）和调试（debugging）

![](image/Pasted%20image%2020260219170005.png)


### 2.6.2 NODE-STYLE HOOKS

![](image/Pasted%20image%2020260219170214.png)

🔹 Node-style hooks（节点式钩子）
* 作为执行流程中的适配器（adapter）参与执行。
* **不能终止执行流程**。
* 但可以修改：
  * prompt
  * 模型
  * 工具
  * 对话历史等内容


![](image/Pasted%20image%2020260219170231.png)

Node-style and wrap-style hooks are called in the order they where introduced to the agent
![](image/Pasted%20image%2020260219170934.png)
### 2.6.3 Wrap-style hooks


![](image/Pasted%20image%2020260219170214.png)


 🔹 Wrap-style hooks（包裹式钩子）
* 是围绕模型或工具调用的执行包装器（execution wrappers）。
* **拥有执行控制权**。
* 可以在模型调用之前和之后执行自定义逻辑。
* 甚至可以跳过模型调用（shortcut），例如：0
  * 当检测到相同 prompt 的缓存结果已经存在时，可以直接返回缓存结果，而不再调用模型。


![](image/Pasted%20image%2020260219170958.png)

---

BYPASS MODEL CALL IN WRAP-STYLE HOOKS

Second wrapper is not called because first wrapper already returns a cached model response.
Does not call handler with request as parameter but instead returns a ModelResponse in form of an AIMessage. For example, to return a cached model response.

![](image/Pasted%20image%2020260219171324.png)



## 2.7 Built-IN Hooks


![](image/Pasted%20image%2020260129103536.png)



# 3 CONTEXT

## 3.1 Terms 


![](image/Pasted%20image%2020260129104208.png)


Context
The term context is defined as any kind of information an agent has access to during its execution, including besides other short-term as well as long-term memories. Model context refers to the information given to an LLM in a prompt for a single inference step. Tool context refers to what a tool can access and produce. Life-cycle Context are the information a hook has access to.

Context Engineering
The art of creating a model context out of the whole context that takes the particularities of the model's context window size and the model's particularities to interpret information in the context window into consideration. A model context is always a subset of information of the context that is well-selected for a single inference step.

Transient vs. Persistent
The model context is transient because it exists just for a single call and is not saved into the agent's state. The tool context is persistent across tools calls because it is saved into the state.

Memory
Short-term memory survives only a single agent run (dynamic runtime context) or thread (cross-conversation context). Long-term memory survives across threads.

---

Context（上下文）
"上下文"指的是 Agent 在执行过程中可以访问的任何信息，包括短期记忆和长期记忆等。

Model context（模型上下文）：
指在一次推理（inference）步骤中，通过 prompt 提供给 LLM 的信息。

Tool context（工具上下文）：
指工具在执行时可以访问和生成的信息。

Life-cycle Context（生命周期上下文）：
指在执行流程中，hook（钩子）能够访问的信息。


---

Context Engineering（上下文工程）

上下文工程是一门艺术：
它的核心是在"全部上下文（whole context）"中，精心挑选出适合当前模型推理的一部分信息，构造成 模型上下文（model context）。

在构建模型上下文时，需要考虑：

模型的 上下文窗口大小（context window size）

模型在理解和处理上下文信息时的特性

因此：

模型上下文永远只是整体上下文的一个子集，
并且是为某一次推理精心挑选的最合适的信息。


---

Transient vs. Persistent（瞬态 vs. 持久）

模型上下文是瞬态的（transient）
因为它只存在于一次模型调用中，不会被保存到 Agent 的状态中。

工具上下文是持久的（persistent）
因为它会被保存到 Agent 的状态中，并在多个工具调用之间持续存在。

---

Memory（记忆）

Short-term memory（短期记忆）
只在单次 Agent 运行期间存在（动态运行时上下文）
或存在于同一个线程中（跨对话上下文）

Long-term memory（长期记忆）
可以跨多个线程长期存在
在不同会话之间仍然保留


## 3.2 Types of context 

![](image/Pasted%20image%2020260219172219.png)


![](image/Pasted%20image%2020260129105221.png)



row2:  fix information, runtime memory : never update 
row3;  agent state , 会变化 
row 4. long time memoery.  store in DB 

## 3.3 CHANGE SYSTEM PROMPT FOR A SINGLE MODEL CALL (STATE)

Watch out: The system prompt customized for a single model call will not be added as a SystemMessage to the conversation history.

Accessing the conversation history of the agent's state
Declares the function to be a special type of wrap_model_call hook that does not control execution but instead should return a customized system prompt.

![](image/Pasted%20image%2020260219172349.png)



## 3.4 PROMPT INJECTION FED BY RUNTIME
Watch out: The conversation history (copy) customized for a single model call (model context) will not replace the conversation history in the agent's state (dynamic runtime context).

![](image/Pasted%20image%2020260129110410.png)

## 3.5 CUSTOMIZED SYSTEM PROMPT FED BY CUSTOM STATE

Watch out: The customized system prompt for a single model call (model context) will not replace the system prompt of the conversation history in the agent's state (dynamic runtime context). However, for the single model call, the customized system prompt "overwrites" the state's system prompt.


Define a customized dynamic runtime context (custom state)
Get the dynamic runtime context from the request object
Make the customized dynamic runtime context known to the agent Initialize the dynamic runtime context at the time of invocation

![](image/Pasted%20image%2020260129110931.png)


最后的 handler for callbakcs 
 with it, in result, you can see how agent handle the LLM 
 
## 3.6 PROMPT INJECTION FED BY CUSTOM STORE

All node-style hooks are capable to update the agent's state by returning a partial state update instead of None.


Watch out: The conversation history (copy) customized for a single model call (model context) will not replace the conversation history in the agent's state (dynamic runtime context).


Create an in-memory store that lives within the thread. Multiple agents can access the store if configured with it.

Create a key/value pair with key "style" and value "formal academic writing tone" in the store's namespace "writing_style"

Access the store via the runtime object and retrieve the desired values.

Register the store with the agent. Otherwise, hooks will have no access to the store within the runtime


![](image/Pasted%20image%2020260129111518.png)


---

langcahin can not modify the agent state 

## 3.7 UPDATE THE AGENT'S STATE WITH BUILD-IN HOOKS

Summarization build-in node-style hook  

Define the summarization build-in node-style hook. It is implemented as a before_model hook.
Model used for the summarization.
Defines the token count threshold at which the summarization will be triggered. The token count considers the whole conversation history.
Defines how much of the latest messages in the conversation history should remain untouched.




![](image/Pasted%20image%2020260129112308.png)

---
## 3.8 READ FROM CUSTOMIZED STORE IN TOOLS

This is can be considered the most basic retrieval augmented generation (RAG) setup implemented as an AI agent.

Create an in-memory store and initialize it with last memory. Retrieve last memory from store
Return last memory as tool result


![](image/Pasted%20image%2020260129112632.png)

![](image/Pasted%20image%2020260219173616.png)

### 3.8.1 WRITE INTO AND READ FROM CUSTOMIZED STORE

Given a key, the tool retrieves the memory from the store. Keeps the memory with its associated key in the store


![](image/Pasted%20image%2020260129112826.png)


![](image/Pasted%20image%2020260219173624.png)


## 3.9 SHARE CUSTOMIZED STORE AMONG DIFFERENT AGENTS

This is can be considered the most basic agent to agent communication setup within the same thread, where the store is used as the communication medium.


![](image/Pasted%20image%2020260129113530.png)


![](image/Pasted%20image%2020260219173649.png)

