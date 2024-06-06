import torch
from transformers import BertConfig, BertForQuestionAnswering,AutoTokenizer
return_dict=False
# 指定模型和配置路径
model_path = "/ibex/project/c2254/ckpts/bert/converted_13"

tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True, trust_remote_code=False
            )

config = BertConfig.from_pretrained(
        model_path,
        )
    
model = BertForQuestionAnswering.from_pretrained(
    model_path)

try:
    inputs = tokenizer("What does the fox do?", "The quick brown fox jumps over the lazy dog.", return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    print("Inference ran successfully.")
except Exception as e:
    print(f"Error during inference: {e}")