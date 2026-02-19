

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

Stateful:  hide the reasoning (lead to the conclude) in the cache, do not provide the whole conversation history.   do not need provide the reasoning path in the stateful mode. 

Build-in tool:  active through api behind the scene 





# 4 Prompt Engineering


# 5 Context Engineering 


![](image/Pasted%20image%2020260205102931.png)


![](image/Pasted%20image%2020260205103955.png)

cluster the question, get the reasoning path. get a representative example, give the example to the prompt 


![](image/Pasted%20image%2020260205104340.png)



# 6 Model Context Protocol

## 6.1 

好的，我们用一个非常具体和现实的例子来理解 **MCP Client** 和 **MCP Server** 的角色。

### 6.1.1 场景：AI 编程助手
假设你正在使用一个类似 **Cursor** 或 **Claude Desktop** 的 AI 编程/写作工具，这个工具集成了 MCP 协议。

---

### 6.1.2 MCP Client（客户端）
**它是什么**：就是你直接与之交互的那个**应用程序本身**。
*   **具体例子**：`Cursor IDE` 或 `Claude Desktop 应用`。
*   **核心职责**：
    1.  **提供用户界面**：你输入问题、查看代码、和 AI 对话的窗口。
    2.  **托管/调用核心 LLM**：它内置或连接了像 Claude-3.5-Sonnet、GPT-4 这样的核心大语言模型。
    3.  **管理"根路径"和权限**：它知道允许 MCP Server 访问你电脑上的哪些文件夹（比如 `~/projects/my_app`，但排除 `~/Documents/personal`）。
    4.  **协调工作流**：接收你的指令，决定何时调用哪个 MCP Server 的工具，并将结果整合后呈现给你。

---

### 6.1.3 MCP Server（服务器）
**它是什么**：一个独立的、提供**特定功能扩展**的后台服务。它不是一个完整的应用，而更像一个"插件引擎"。
*   **具体例子**：
    *   **`mcp-server-filesystem`**：一个提供文件系统读写权限的服务器。
    *   **`mcp-server-git`**：一个提供 Git 操作（如提交、查看历史、创建分支）的服务器。
    *   **`mcp-server-sqlite`**：一个允许你查询本地 SQLite 数据库的服务器。
    *   **`mcp-server-weather`**：一个可以查询天气 API 的服务器。

---

### 6.1.4 一次完整的交互流程示例

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

### 6.1.5 关键类比与总结

| 概念 | 类比 | 现实例子 |
| :--- | :--- | :--- |
| **MCP Client** | **"主控台"或"指挥中心"** | **Cursor IDE**, **Claude Desktop**, 未来可能整合的 **VS Code**, **Obsidian** 等。 |
| **MCP Server** | **"专用工具箱"或"外挂模块"** | **文件工具包**、**Git工具包**、**数据库连接器**、**Jira 查询器**、**天气插件**等。 |
| **MCP 协议本身** | **"工具箱的连接标准"** | 就像 USB-C 接口，让任何符合标准的"工具箱"（Server）都能插到"主控台"（Client）上使用。 |

**核心价值**：
*   **安全可控**：Client 控制权限（Roots）。Git 服务器只能操作 Git，不能乱删你的文件。
*   **模块化与生态**：开发者可以轻松创建新的 Server（如连接公司内部 API 的 Server），任何支持 MCP 的 Client 都能立即使用它，无需等待 Client 自身更新。
*   **体验统一**：你可以在不同的 Client（如 Cursor 或 Claude Desktop）中使用同一套熟悉的工具（Servers），获得一致的 AI 增强体验。


## 6.2 Initialization Phase 

![](image/Pasted%20image%2020260205110528.png)


![](image/Pasted%20image%2020260205110624.png)






