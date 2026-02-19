

# 1 Pre-Trainning

![[images/Pasted image 20251120105121.png]]



![[images/Pasted image 20251120105610.png]]


max length of sequence  can be allowed    is the the size of content window 



batch ? 
sequence ?   content size in one  windows ? 

content size  is denfind by LLM,   key value matrix 决定了 content size 
each sequence in a batch is in the same size 


![[images/Pasted image 20251120110709.png]]


![[images/Pasted image 20251120111013.png]]


Gridient vanishing:  the input hat no influence on the last layer  

Gridient Explode: a little change in input has huge impact on the output 




![[images/Pasted image 20251120113142.png]]


![[images/Pasted image 20251120113611.png]]

gradient: find the direct where the loss change at most 

![[images/Pasted image 20251120114329.png]]


