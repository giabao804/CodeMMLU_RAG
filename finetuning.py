from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    Trainer
)
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from datasets import Dataset
import torch
from torch import nn 

# model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )

class QwenForMCQA(nn.Module):
    def __init__(self, model):
        super(QwenForMCQA, self).__init__()
        self.model = model
        
        # Add new layers after the original model
        self.classifier = nn.Sequential(
            nn.Linear(model.config.hidden_size, 512),  # First layer of the new head
            nn.ReLU(),
            nn.Linear(512, 4)  # 4 output classes for MCQA (adjust as necessary)
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Get the model's output
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        
        # Use only the last token representation for classification (assuming it's a sentence-level task)
        logits = self.classifier(hidden_states[:, -1, :])
        return logits



from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# print methods in tokenizer 

print(dir(tokenizer))

# print input list in tokenizer.__init__()

print(tokenizer.__init__.__code__.co_varnames)
print(tokenizer.__init__.__code__.co_argcount)

# check code tokenizer 










