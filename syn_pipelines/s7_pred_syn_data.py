# %%
# %%
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

# if you want to run through jupyter, please set it as false.
import sys
sys.path.append("../")
cmd_args = False
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--file_dir', default='<YOUR_DATASET_DIR>', help='file name')
parser.add_argument('--file_name', default='conll_2003', help='file name of the dataset, you should make sure it contains `train_1000.jsonl` file')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--from_pretrained', default='<PLM_PATH>', help='model from pretrained')
parser.add_argument('--model_from_pretrained', default='<PLM_PATH>', help='model from pretrained')
parser.add_argument('--dense_lang', default='0', help='whether the language is character-dense language')
parser.add_argument('--batch_size', default=4, help='batch size')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

from transformers import BertTokenizer, BertConfig
from main.predictor.knowfree_predictor import KnowFREEPredictor

DA_DIR = os.path.join(args.file_dir, args.file_name + f'_{args.save_type_name}_DA')
LABEL_FILE = os.path.join(DA_DIR, 'labels.txt')
SOURCE_FILE = os.path.join(DA_DIR, "1000", 'train_1000_synthetic.jsonl')

tokenizer = BertTokenizer.from_pretrained(args.from_pretrained)
config = BertConfig.from_pretrained(args.from_pretrained)
pred = KnowFREEPredictor(tokenizer=tokenizer, config=config, from_pretrained=args.model_from_pretrained,
                          label_file=LABEL_FILE, batch_size=args.batch_size)

# %%
with open(SOURCE_FILE) as f:
    ori_data = f.readlines()
data = [json.loads(i) for i in ori_data]
if str(args.dense_lang) == '1':
    data_text = [''.join(i['text']) for i in data]
else:
    data_text = [i['text'] for i in data]
entities_list = []

for entities in pred(data_text):
    entities_list.extend(entities)

# %%
for item, ext_entities in zip(data, entities_list):
    for entity in ext_entities:
        if entity not in item['entities']:
            item['entities'].append(entity)

split_list = [250, 500, 1000]
for size in split_list:
    SAVE_FILE = os.path.join(DA_DIR, str(size), f'train_{str(size)}_synthetic.jsonl')
    sample_size = int(size / 1000 * len(ori_data))
    sample_data = data[:sample_size]
    with open(SAVE_FILE, 'w') as f:
        for i in sample_data:
            f.write(json.dumps(i, ensure_ascii=False) + '\n')

# %%
