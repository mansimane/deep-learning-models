from transformers import DebertaTokenizer, DebertaModel
import torch
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

model = DebertaModel.from_pretrained('microsoft/deberta-base')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)