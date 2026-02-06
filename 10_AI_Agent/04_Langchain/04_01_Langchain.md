

![](image/Pasted%20image%2020260115102020.png)


![](image/Pasted%20image%2020260115102850.png)

# 1 Structured output 


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


Are Agents Black Boxes?
An agents is invoked, executes and provides a result. However, the execution may involve multiple agentic loops.
LangChain's provides options to "look into" agents during execution and to run custom code at specific points in the agent's execution flow:
Core agentic loop
- Handlers (for callbacks): Follows observer pattern and do not own the execution of, e.g., the model. Used primarily to log, trace and debug.
- Middleware (node-style and wrap-style hooks): wrap-style hooks are execution wrappers around the model or tool calls. They control the execution. They allow to execute logic before and after the model call and they can even shortcut the model call if the model is not needed, e.g., because of existing cached results from former model calls with same prompt. Node-style hooks are adapters that participate in execution. They can't stop the execution flow but are allowed to change prompts, models, tools, even the conversation history, etc..

![](image/Pasted%20image%2020260129102053.png)

---


how to bypass the model call
like you have already asked the same question. then next time with the same example, we can pass the modle call 

![](image/Pasted%20image%2020260129103316.png)



## 2.1 BuiltIn Hooks

![](image/Pasted%20image%2020260129103536.png)



---

![](image/Pasted%20image%2020260129104208.png)

---

![](image/Pasted%20image%2020260129105221.png)



row2:  fix information, runtime memory : never update 
row3;  agent state , 会变化 
row 4. long time memoery.  store in DB 

![](image/Pasted%20image%2020260129110410.png)


![](image/Pasted%20image%2020260129110651.png)

最后的 handler for callbakcs 
 with it, in result, you can see how agent handle the LLM 
 
![](image/Pasted%20image%2020260129110931.png)


![](image/Pasted%20image%2020260129111518.png)


---

langcahin can not modify the agent state 



----

Summarization build-in node-style hook  

![](image/Pasted%20image%2020260129112308.png)

---

read from 

![](image/Pasted%20image%2020260129112632.png)

----

![](image/Pasted%20image%2020260129112826.png)


![](image/Pasted%20image%2020260129113530.png)




