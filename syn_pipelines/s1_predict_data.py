# %%
import os
from argparse import ArgumentParser

# if you want to run through jupyter, please set it as false.
import sys
sys.path.append("../")
cmd_args = False
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=0, help='n_gpu')
parser.add_argument('--skip', default=-1, help='skip the first n lines, the skip index is count from the start index of n-th chunks')
parser.add_argument('--file_dir', default='<YOUR_DATASET_DIR>', help='file name')
parser.add_argument('--file_name', default='conll_2003', help='file name of the dataset, you should make sure it contains `train_1000.jsonl` file')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--model_from_pretrained', default='<LLM_PATH>', help='model from pretrained')
parser.add_argument('--batch_size', default=20, help='batch size')
parser.add_argument('--dense_lang', default='0', help='whether the language is character-dense language')
parser.add_argument('--mode', default='pos', help='predict entity or pos')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)
MODE = args.mode

import json
import random
from tqdm import tqdm

if args.save_type_name == 'GLM3':
    from llm.chatglm import Predictor
else:
    from llm.llm import Predictor

SOURCE_FILE = os.path.join(args.file_dir, args.file_name, 'train_1000.jsonl')
SAVE_DIR = os.path.dirname(SOURCE_FILE) + f'_{args.save_type_name}_DA'
basename = os.path.basename(SOURCE_FILE)
pred = Predictor(model_from_pretrained=args.model_from_pretrained)

prompt_dict = {'entity': '''指令: 请识别并抽取输入句子的命名实体，并使用JSON格式的数组进行返回，子项包括entity和type属性：
格式要求: 1. 输出格式为[{entity: '', type: ''}],其中entity表示所提取的实体文本, type表示所提取的实体类型, 一个entity对应一个type
2. 如果不存在任何实体，请输出空数组[]
输入: ''',
'pos': '''指令: 请提取输入句子的词性(POS)，并使用JSON格式的数组进行返回，子项包括word和pos属性：
格式要求: 1. 输出格式为[{word: '', pos: ''}],其中word表示所提取的文本, pos表示所提取的词性, 一个word对应一个pos
2. 请务必将输入中**所有字符**和**标点**都进行标注
输入: '''}
with open(SOURCE_FILE) as f:
    ori_data = f.readlines()

ori_data = [json.loads(i) for i in ori_data]
selected_data = ori_data[int(args.skip):] if int(args.skip) > -1 else ori_data
data = []

SPLIT_TAG = '' if str(args.dense_lang) == '1' else ' '
for item in tqdm(ori_data):
    text = item['text']
    text = SPLIT_TAG.join(text)
    user_content = prompt_dict[MODE] + text
    data.append((user_content, text))

num_batches = len(data) // args.batch_size + 1 if len(data) % args.batch_size != 0 else len(data) // args.batch_size

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def denoise(ori):
    ori = ori.replace('```json', '')
    ori = ori.replace('\n', '')
    ori = ori.replace('```', '')
    return ori

for i in tqdm(range(num_batches)):
    max_length = 0
    samples = data[i * args.batch_size : (i + 1) * args.batch_size]
    oris = [item[1] for item in samples]
    inputs = []
    for item in samples:
        inputs.append(item[0])
        if len(item[0]) > max_length:
            max_length = len(item[0])
    outputs = pred(inputs, max_new_tokens=5*max_length, temperature=0.8, build_message=True)
    with open(os.path.join(SAVE_DIR, f'{MODE}_{basename}'), 'a', encoding='utf-8') as f:
        for ori, out in zip(oris, outputs):
            f.write('{}\t{}\n'.format(ori, denoise(out)))

# %%
