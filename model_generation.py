import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForQuestionAnswering
import json

def load_model_and_tokenizer(config_dir):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(config_dir, padding_side='left', use_fast=False, trust_remote_code=True)
    tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>"
    })
    model = AutoModelForCausalLM.from_pretrained(config_dir, low_cpu_mem_usage=True, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def prepare_inputs(tokenizer, context, question):
    """准备模型输入"""
    return tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)

def generate_answer(model, tokenizer, inputs):
    """生成答案"""
    input_ids = inputs["input_ids"].long()
    outputs = model(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def load_dataset(file_path):
    """加载数据集"""
    with open(file_path, 'r') as file:
        return json.load(file)['data']

def make_predictions(model, tokenizer, dataset):
    predictions = {}
    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # 准备输入，使用编码的方法
                input_ids = tokenizer.encode(question + " " + context, return_tensors="pt")
                # 生成答案
                outputs = model.generate(input_ids, max_length=50)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions[qa['id']] = answer
    return predictions

def save_predictions(predictions, file_path):
    """保存预测到文件"""
    with open(file_path, 'w') as file:
        json.dump(predictions, file)
    print("Predictions saved to", file_path)

# 主程序
if __name__ == '__main__':
    config_dir = '/home/jingwel/llm/Arabic-NLP/eval/qa_context/model/hf-ep3-ba655000'
    dataset_file = '/home/jingwel/llm/Arabic-NLP/eval/qa_context/data/test-context-ar-question-ar.json'
    prediction_file = '/home/jingwel/llm/Arabic-NLP/eval/qa_context/data/prediction_file_ba655000.json'
    
    model, tokenizer = load_model_and_tokenizer(config_dir)
    dataset = load_dataset(dataset_file)
    predictions = make_predictions(model, tokenizer, dataset)
    save_predictions(predictions, prediction_file)
