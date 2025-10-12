# %%
import os
import re
import json
import random
import json_repair
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# if you want to run through jupyter, please set it as false.
import sys
sys.path.append("../")
cmd_args = False
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=1, help='n_gpu')
parser.add_argument('--file_dir', default='<YOUR_DATASET_DIR>', help='file name')
parser.add_argument('--file_name', default='conll_2003', help='file name of the dataset, you should make sure it contains `train_1000.jsonl` file')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--model_from_pretrained', default='<LLM_PATH>', help='model from pretrained')
parser.add_argument('--dense_lang', default='0', help='whether the language is character-dense language')
parser.add_argument('--batch_size', default=40, help='batch size')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

if args.save_type_name == 'GLM3':
    from llm.chatglm import Predictor
else:
    from llm.llm import Predictor

pred = Predictor(model_from_pretrained=args.model_from_pretrained)

SOURCE_FILE = os.path.join(args.file_dir, args.file_name, 'train_1000.jsonl')
SAVE_DIR = os.path.dirname(SOURCE_FILE) + f'_{args.save_type_name}_DA'
basename = os.path.basename(SOURCE_FILE)
save_name = basename.split('.')[0] +'_synthetic.jsonl'
SAVE_FILE = os.path.join(SAVE_DIR,
                         save_name)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

BATCH_SIZE = args.batch_size
CLIP = True
with open(SOURCE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()
ori_data = [json.loads(i) for i in ori_data]

if str(args.dense_lang) == '1':
    prompt_prefix = '''指令: 你作为拥有丰富知识储备的专家，需要根据我给出的样例进行续写，给学生们解释样例中包含的实体含义，我会给定你样例和其包含的实体加实体类型，请续写样例的后文，并解释实体的含义。
样例：'''
else:
    prompt_prefix = '''Instruction: Please act a knowledgeable expert, you are required to continue the given sample and explain the meaning of the entities it contains to students. I will provide you with a sample along with the entities it includes and their corresponding entity types. Please continue the sample and explain the meanings of the entities.
sample:'''

SPLIT_TAG = '' if str(args.dense_lang) == '1' else ' '

# %%
ask_list = []
for idx, ori_item in tqdm(enumerate(ori_data), total=len(ori_data)):
    text = SPLIT_TAG.join(ori_item['text'])
    ori_entities = [SPLIT_TAG.join(item['text']) + '({})'.format(item['entity']) for item in ori_item["entities"]]
    if len(ori_entities) == 0:
        continue
    ask_content = prompt_prefix + text + \
            '\n\n实体：' + '、'.join(ori_entities) + '\n输出：\n'
    ask_list.append(ask_content)

num_batches = len(ask_list) // BATCH_SIZE + 1 if len(ask_list) % BATCH_SIZE != 0 else len(ask_list) // BATCH_SIZE
for i in tqdm(range(num_batches)):
    sample = ask_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
    outputs = pred(sample, max_new_tokens=len(sample[0]), temperature=0.8, build_message=True)
    for res in outputs:
        res = res.replace('\n', '')
        if str(args.dense_lang) == '1':
            res = res.replace(' ', '')
        res = res[:256]
        if CLIP:
            res = re.split(r'[。.]', res)
            for r in res:
                if len(r) == 0:
                    continue
                if str(args.dense_lang) == '1':
                    res_text = list(r)
                else:
                    res_text = re.findall(r'\w+|\S', r)
                with open(SAVE_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'text': res_text,
                        'entities': [],
                        'synthetic': True
                    }, ensure_ascii=False) + '\n')
        else:
            with open(SAVE_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'text': list(res),
                    'entities': [],
                    'synthetic': True
                }, ensure_ascii=False) + '\n')

# %%
