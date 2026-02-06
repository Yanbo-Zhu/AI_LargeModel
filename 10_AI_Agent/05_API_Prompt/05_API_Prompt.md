

# 1 Prompt Format 



# 2 API 

Stateful:  hide the reasoning (lead to the conclude) in the cache, do not provide the whole conversation history.   do not need provide the reasoning path in the stateful mode. 

Build-in tool:  active through api behind the scene 





# 3 Prompt Engineering


# 4 Context Engineering 


![](image/Pasted%20image%2020260205102931.png)


![](image/Pasted%20image%2020260205103955.png)

cluster the question, get the reasoning path. get a representative example, give the example to the prompt 


![](image/Pasted%20image%2020260205104340.png)



# 5 Model Context Protocol

## 5.1 

好的，我们用一个非常具体和现实的例子来理解 **MCP Client** 和 **MCP Server** 的角色。

### 5.1.1 场景：AI 编程助手
假设你正在使用一个类似 **Cursor** 或 **Claude Desktop** 的 AI 编程/写作工具，这个工具集成了 MCP 协议。

---

### 5.1.2 MCP Client（客户端）
**它是什么**：就是你直接与之交互的那个**应用程序本身**。
*   **具体例子**：`Cursor IDE` 或 `Claude Desktop 应用`。
*   **核心职责**：
    1.  **提供用户界面**：你输入问题、查看代码、和 AI 对话的窗口。
    2.  **托管/调用核心 LLM**：它内置或连接了像 Claude-3.5-Sonnet、GPT-4 这样的核心大语言模型。
    3.  **管理"根路径"和权限**：它知道允许 MCP Server 访问你电脑上的哪些文件夹（比如 `~/projects/my_app`，但排除 `~/Documents/personal`）。
    4.  **协调工作流**：接收你的指令，决定何时调用哪个 MCP Server 的工具，并将结果整合后呈现给你。

---

### 5.1.3 MCP Server（服务器）
**它是什么**：一个独立的、提供**特定功能扩展**的后台服务。它不是一个完整的应用，而更像一个"插件引擎"。
*   **具体例子**：
    *   **`mcp-server-filesystem`**：一个提供文件系统读写权限的服务器。
    *   **`mcp-server-git`**：一个提供 Git 操作（如提交、查看历史、创建分支）的服务器。
    *   **`mcp-server-sqlite`**：一个允许你查询本地 SQLite 数据库的服务器。
    *   **`mcp-server-weather`**：一个可以查询天气 API 的服务器。

---

### 5.1.4 一次完整的交互流程示例

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

### 5.1.5 关键类比与总结

| 概念 | 类比 | 现实例子 |
| :--- | :--- | :--- |
| **MCP Client** | **"主控台"或"指挥中心"** | **Cursor IDE**, **Claude Desktop**, 未来可能整合的 **VS Code**, **Obsidian** 等。 |
| **MCP Server** | **"专用工具箱"或"外挂模块"** | **文件工具包**、**Git工具包**、**数据库连接器**、**Jira 查询器**、**天气插件**等。 |
| **MCP 协议本身** | **"工具箱的连接标准"** | 就像 USB-C 接口，让任何符合标准的"工具箱"（Server）都能插到"主控台"（Client）上使用。 |

**核心价值**：
*   **安全可控**：Client 控制权限（Roots）。Git 服务器只能操作 Git，不能乱删你的文件。
*   **模块化与生态**：开发者可以轻松创建新的 Server（如连接公司内部 API 的 Server），任何支持 MCP 的 Client 都能立即使用它，无需等待 Client 自身更新。
*   **体验统一**：你可以在不同的 Client（如 Cursor 或 Claude Desktop）中使用同一套熟悉的工具（Servers），获得一致的 AI 增强体验。


## 5.2 Initialization Phase 

![](image/Pasted%20image%2020260205110528.png)


![](image/Pasted%20image%2020260205110624.png)






