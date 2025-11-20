

# 1 Pre-Trainning

![[Pasted image 20251120105121.png]]



![[Pasted image 20251120105610.png]]


max length of sequence  can be allowed    is the the size of content window 



batch ? 
sequence ?   content size in one  windows ? 

content size  is denfind by LLM,   key value matrix 决定了 content size 
each sequence in a batch is in the same size 


![[Pasted image 20251120110709.png]]


![[Pasted image 20251120111013.png]]


Gridient vanishing:  the input hat no influence on the last layer  

Gridient Explode: a little change in input has huge impact on the output 




![[Pasted image 20251120113142.png]]


![[Pasted image 20251120113611.png]]

gradient: find the direct where the loss change at most 

![[Pasted image 20251120114329.png]]


