import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import ast


model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)


embedding_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
embedding_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')