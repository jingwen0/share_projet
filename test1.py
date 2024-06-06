import json

with open('./data/validation/dev-context-ar-question-ar.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

converted_data = []

for item in data['data']:
    for paragraph in item['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            converted_item = {
                'id': qa['id'],
                'title': item['title'],
                'context': context,
                'question': qa['question'],
                'answers': {
                    'text': [answer['text'] for answer in qa['answers']],
                    'answer_start': [answer['answer_start'] for answer in qa['answers']]
                }
            }
            converted_data.append(converted_item)

# 保存转换后的数据到新的JSON文件
with open('./data/converted_dev-context-ar-question-ar.json', 'w', encoding='utf-8') as file:
    json.dump(converted_data, file, ensure_ascii=False, indent=2)
