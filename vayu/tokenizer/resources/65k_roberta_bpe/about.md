This is a Byte Pair Encoding tokenizer similar to that of GPT-2 trained using accompanying script.

Vocab size is: `65536` tokens

Trained on 7 CPT raw corpus of 500M+ sentences (non-lowercase, non-stemmed, non-normalized).

Also included `config.json` that can be used to configure a Roberta model for training language models.

This is trained using Rust based tokenizers library and should be loaded using the "Fast" version of transformers
 tokenizer class 
```
from transformers import RobertaTokenizerFast 
tokenizer = RobertaTokenizerFast.from_pretrained("./deep_learning/tokenizer/resources/65k_roberta/") 

In [10]: tokenizer.encode('how are you ?')                 
Out[10]: [0, 6039, 513, 698, 739, 2]

In [11]: tokenizer.encode(['how are you ?'])    
Out[11]: [0, 6039, 513, 698, 739, 2]
```