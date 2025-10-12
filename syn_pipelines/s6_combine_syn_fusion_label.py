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

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

label_dict = {
    'OTHER': ['OTHER', 'other']
}

DA_DIR = os.path.join(args.file_dir, args.file_name + f'_{args.save_type_name}_DA')

result = {}

with open(os.path.join(DA_DIR, 'label_fusion.json')) as f:
    label_fusion = json.load(f)

with open(os.path.join(DA_DIR, 'label_syn_fusion.json')) as f:
    label_syn_fusion = json.load(f)

for key in label_fusion:
    result[key] = label_fusion[key]

for key in label_syn_fusion:
    if key not in result:
        result[key] = label_syn_fusion[key]
    else:
        result[key]['count'] += label_syn_fusion[key]['count']

count = 0
for key in result:
    result[key]['idx'] = count
    count += 1

with open(os.path.join(DA_DIR, 'label_syn_fusion.json'), mode='w+') as f:
    json.dump(result, f)

# %%
