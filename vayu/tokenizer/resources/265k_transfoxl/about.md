This is original transformerXL tokenizer
* not trained on CDR corpus
* vocabulary of ~265k tokens
* does not used sub word tokenization
* should be loaded using the "Fast" version of transformers tokenizer class 
```
from transformers import TransfoXLTokenizerFast, 
tokenizer = TransfoXLTokenizerFast.from_pretrained('transfo-xl-wt103')  

In [13]: tokenizer.encode('how are you ?')
Using pad_token, but it is not set yet.
Out[13]: [433, 37, 304, 788]

In [14]: tokenizer.encode(['how are you ?']) 
Using pad_token, but it is not set yet.
Out[14]: [433, 37, 304, 788]
```