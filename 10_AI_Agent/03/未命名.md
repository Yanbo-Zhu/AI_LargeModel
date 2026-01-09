
# 1 Fine Tunning




![[image/Pasted image 20251204104600.png]]



![[image/Pasted image 20251204105651.png]]








# 2 Distillation


![[image/Pasted image 20251204110823.png]]

![[image/Pasted image 20251211101902.png]]


OpenAI only give 20 log probalities 


---

# 3 Quantization 

![[image/Pasted image 20251211102827.png]]


![[image/Pasted image 20251211103203.png]]

Format: put the quantization res 
all scaling the 


# 4 Benchmarking 

![[image/Pasted image 20251211103559.png]]


fast evolution of LLMs:    metrics hast range, when LLm evolves.  the range shrickn and reach 100% good proformance .   the task is too easy for the evolved LLM 



Prompt sensitivity and contamination 


![[image/Pasted image 20251211104519.png]]


![[image/Pasted image 20251211105120.png]]



how cloud we jedget the result without reference 
- check the length of oiutputs, check formular , check the the meaning of context 



![[image/Pasted image 20251211105249.png]]


![[image/Pasted image 20251211105436.png]]

test the LLM,  or the ai system contains the LLM 


statisitcla reliabilty: fewer entry
low human subjectibly: human decide if it is in fact or not 


![[image/Pasted image 20251211105820.png]]

F1 Source : both the precision and recall , thier weight is same  

$(1+beta^2)  multiple ((P x R) / (beta^2 mal P + R))$  
beta is wight more, give more focus to xx  . beata > 1 , give focus on recall


ranking metrics

K items 
give many of list to LLM and give the k top relavant    


1 ist very good ranking
0 ist very bad ranking:   0 means there are 0 relevants tieims 

(1/T) x (sum 1/R)

T: number of test
R: ranking betwenn 1 and 0 



![[image/Pasted image 20251211111138.png]]


BLEU:  refenence and output    n-gram also  existed in reference .  结合 Panelity 使用 


ROUGHE-n:   regarding the order,   only word , not semenatic meaning conprision

![[image/Pasted image 20251211111949.png]]



![[image/Pasted image 20251211112320.png]]

How consistent: how often the response simliar if i give the same prompt 

the more similar they are, the more consistent is my LLM

![[image/Pasted image 20251211112524.png]]

![[image/Pasted image 20251211112741.png]]


![[image/Pasted image 20251211112848.png]]


![[image/Pasted image 20251211112914.png]]


![[image/Pasted image 20251211113116.png]]


![[image/Pasted image 20251211113313.png]]

![[image/Pasted image 20251211113352.png]]

 how good LLM resolve the git issue 

![[image/Pasted image 20251211113616.png]]


![[image/Pasted image 20251211113646.png]]

Prompt 
instction\_id\_list:   condition   to excute the test 
Kwargs: 


![[image/Pasted image 20251211114039.png]]


compare the LLm's solution without the musterlosung in buch



































































