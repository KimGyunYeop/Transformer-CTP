# Transformer-CTP

This code for Transformer-CTP: Current Token Prediction Using Cross-Attention of Queries with Current Position Information  
modeling_*.py is reference from modeling_bart.py from huggingface.transformer  
We made some modifications to modeling.py to make our experiment work.  
To easily see the modifications we made, search for "#code for proposed methods" in modeling_mc.py  
The other modeling_*.py is almost the same as modeling_mc.py.  


### Usage  
this example is experiment in WMT'16 En->De dataset  

#### make tokenizer  
``` 
python construct_tokenizer.py \
  --data_name wmt14 \
  --subset de-en \
  --src_lang en \
  --tgt_lang de
```

#### train from scratch with paper's setting
``` 
python -u train_mixcoder.py \
        --data_name wmt14 \
        --subset de-en \
        --src_lang en \
        --tgt_lang de \
        --next_token_type {new_token, avg_prev_token} \
        --setting 2
```

#### train using pre-trained weight(BART-base) with paper's setting
``` 
python -u train_mixcoder_for_ptbart.py \
        --data_name wmt14 \
        --subset de-en \
        --src_lang en \
        --tgt_lang de \
        --next_token_type {new_token, avg_prev_token} \
        --setting 1
```
