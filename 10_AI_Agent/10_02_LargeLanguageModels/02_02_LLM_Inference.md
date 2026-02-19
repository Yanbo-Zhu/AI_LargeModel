

# 1 Embeddings 

positional encoding:
1. be unique
2. show the realtionship of each of position. The distance of two differnt position 



![[images/Pasted image 20251113104433.png]]


# 2 Attention 

what kind of words influeces the current of words right now 

![[images/Pasted image 20251113105417.png]]


![[images/Pasted image 20251113110115.png]]



Masking: we only use the post token influcence the cureent token  
![[images/Pasted image 20251113110625.png]]



![[images/Pasted image 20251113110810.png]]



serfisticate 


# 3 multi-head self-attention 




4 MLP 

i b

# 4 Output 

![[images/Pasted image 20251120103630.png]]



![[images/Pasted image 20251120104128.png]]



![[images/Pasted image 20251120104430.png]]


the final embedding (it accumalte all the meaning of the prvious words). this embedding is a konzept
demensinal of embidding vectorr:   demsionmen of d_model  , d_model is the matrix 

linaer projection:  understanding the embeding,  tanslate it into all possibele konzept (each konzept hase a score value)

softmax: scale it into `[0,1]`

sampling: strategy how to generate the disturtion , sue this distrition help use to generate the new words 



## 4.1 temperature  in output processing 


你说的 “tempatur” 多半是 temperature（温度），它不是 Transformer 架构本身的组成部分，而是 在 Transformer 生成文本（如 GPT、T5 Decoder 等）时用于控制输出随机性的一个超参数


![[images/Pasted image 20251120104946.png]]





# 5 Experiment 

![[images/Pasted image 20251120104533.png]]












