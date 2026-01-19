

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











