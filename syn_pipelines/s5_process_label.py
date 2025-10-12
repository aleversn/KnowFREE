import re
import json
import json_repair
from tqdm import tqdm
from argparse import ArgumentParser

# if you want to run through jupyter, please set it as false.
import sys
sys.path.append("../")
cmd_args = False
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--file_dir', default='<YOUR_DATASET_DIR>', help='file name')
parser.add_argument('--file_name', default='mit_movies', help='file name of the dataset, you should make sure it contains `train_1000.jsonl` file')
parser.add_argument('--is_syn', default=1, help='is process synthetic file')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--dense_lang', default='0', help='whether the language is character-dense language')
parser.add_argument('--label_prefix', default='', help='label prefix')
parser.add_argument('--entity_label', default='<DIR_NAME>/entity_label.json', help='label format')
parser.add_argument('--pos_label', default='<DIR_NAME>/pos_label.json', help='label format')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

label_dict = {
    'OTHER': ['OTHER', 'other']
}

count_dict = {}
except_dict = {}

SPLIT_TAG = '' if str(args.dense_lang) == '1' else ' '


def update_label_dict(file_name):
    with open(file_name) as f:
        e_dict = json_repair.load(f)
    for key in e_dict:
        cur = e_dict[key]
        if key not in cur:
            cur.append(key)
        if key not in label_dict:
            label_dict[key] = cur
        else:
            label_dict[key] += cur

update_label_dict(args.entity_label)
update_label_dict(args.pos_label)

for key in label_dict:
    cur = label_dict[key]
    new_cur = []
    for item in cur:
        if item not in new_cur:
            new_cur.append(item)
    label_dict[key] = new_cur

label_format_dict = {}

for key in label_dict:
    for key_item in label_dict[key]:
        label_format_dict[key_item] = key

def label_format(key, distinct_pos=False, prefix=args.label_prefix):
    key = key.strip()
    if distinct_pos and key in ['OTHER']:
        if 'POS' not in count_dict:
            count_dict['POS'] = 1
        else:
            count_dict['POS'] += 1
        return 'POS'
    if key in label_format_dict:
        format_key = label_format_dict[key]
        format_key = prefix + format_key
        if format_key not in count_dict:
            count_dict[format_key] = 1
        else:
            count_dict[format_key] += 1
        return format_key
    count_dict[prefix + 'OTHER'] += 1
    if key not in except_dict:
        except_dict[key] = 1
    else:
        except_dict[key] += 1
    return 'WORD'

# %%
label_format('OTHER')

# %%
import os
from copy import deepcopy
SOURCE_FILE = os.path.join(args.file_dir, args.file_name, 'labels.txt')
DA_DIR = os.path.join(args.file_dir, args.file_name + f'_{args.save_type_name}_DA')
LABEL_FILE = SOURCE_FILE
if int(args.is_syn) == 0:
    ORI_FILE = os.path.join(os.path.dirname(SOURCE_FILE), 'train_1000.jsonl')
    EXT_ENTITY_FILE = os.path.join(DA_DIR, '1000', 'entity_train_1000.jsonl')
    EXT_POS_FILE = os.path.join(DA_DIR, '1000', 'pos_train_1000.jsonl')
    SAVE_LABEL_FILE = os.path.join(DA_DIR, 'label_fusion.json')
else:
    ORI_FILE = os.path.join(DA_DIR, '1000', 'train_1000_synthetic.jsonl')
    EXT_ENTITY_FILE = os.path.join(DA_DIR, '1000', 'entity_train_1000_synthetic.jsonl')
    EXT_POS_FILE = os.path.join(DA_DIR, '1000', 'pos_train_1000_synthetic.jsonl')
    SAVE_LABEL_FILE = os.path.join(DA_DIR, 'label_syn_fusion.json')
DISABLED_ORI_LABELS = False
IGNORE_LABELS = []
# IGNORE_LABELS = ['ADVERB', 'NOUN', 'PROPER_NOUN', 'ADJECTIVE', 'QUANTIFIER']

# %%
with open(LABEL_FILE) as f:
    ori_labels = f.readlines()
ori_labels = [i.strip() for i in ori_labels]
with open(ORI_FILE) as f:
    ori_data = f.readlines()
ori_data = [json_repair.loads(i) for i in ori_data]
ori_data_copy = deepcopy(ori_data)
if DISABLED_ORI_LABELS:
    for item in ori_data:
        item['entities'] = []
        item['mask_ori'] = True

dataset_fusion_labels = []

# %%
with open(EXT_ENTITY_FILE, encoding='utf-8', mode='r') as f:
    ori_list = f.readlines()
for idx, item in enumerate(tqdm(ori_list)):
    item = item.split('\t')
    json_item = json_repair.loads(item[1])
    exists_2d = {}
    if not isinstance(json_item, list):
        continue
    for item in json_item:
        if type(item) != dict or 'entity' not in item or 'type' not in item: continue
        entity, entity_type = str(item['entity']), item['type']
        if str(args.dense_lang) == '1':
            ent_len = len(entity)
        else:
            ent_len = len(re.findall(r'\w+|\S', entity))
        if ent_len <= 1:
            continue
        entity_type = label_format(entity_type)
        if entity_type in IGNORE_LABELS:
            continue
        if entity_type not in dataset_fusion_labels:
            dataset_fusion_labels.append(entity_type)
        ori_text_list = ori_data[idx]['text']
        for i in range(len(ori_text_list) - ent_len + 1):
            if SPLIT_TAG.join(ori_text_list[i:i+ent_len]) == entity:
                if i not in exists_2d:
                    exists_2d[i] = {}
                    if (i + ent_len) not in exists_2d[i]:
                        ori_data[idx]['entities'].append({'start': i, 'end': i+ent_len, 'entity': entity_type, 'text': ori_text_list[i:i+ent_len]})
                        exists_2d[i][i+ent_len] = 1

# %%
with open(EXT_POS_FILE, encoding='utf-8', mode='r') as f:
    ori_list = f.readlines()
for idx, item in enumerate(tqdm(ori_list)):
    item = item.split('\t')
    json_item = json_repair.loads(item[1])
    exists_2d = {}
    if not isinstance(json_item, list):
        continue
    for item in json_item:
        if type(item) != dict or 'word' not in item or 'pos' not in item: continue
        entity, entity_type = str(item['word']), item['pos']
        if str(args.dense_lang) == '1':
            ent_len = len(entity)
        else:
            ent_len = len(re.findall(r'\w+|\S', entity))
        if ent_len <= 1:
            continue
        entity_type = label_format(entity_type)
        if entity_type in IGNORE_LABELS:
            continue
        if entity_type not in dataset_fusion_labels:
            dataset_fusion_labels.append(entity_type)
        ori_text_list = ori_data[idx]['text']
        for i in range(len(ori_text_list) - ent_len + 1):
            if SPLIT_TAG.join(ori_text_list[i:i+ent_len]) == entity:
                if i not in exists_2d:
                    exists_2d[i] = {}
                    if (i + ent_len) not in exists_2d[i]:
                        ori_data[idx]['entities'].append({'start': i, 'end': i+ent_len, 'entity': entity_type, 'text': ori_text_list[i:i+ent_len]})
                        # ori_data[idx]['entities'].append({'start': i, 'end': i+ent_len, 'entity': 'WORD', 'text': ori_text_list[i:i+ent_len]})
                        exists_2d[i][i+ent_len] = 1

# %%
split_list = [250, 500, 1000]
for size in split_list:
    SAVE_FILE = os.path.join(DA_DIR, str(size), os.path.splitext(os.path.basename(ORI_FILE))[0].replace('1000', str(size)) + '_fusion{}.jsonl'.format('_mask' if DISABLED_ORI_LABELS else ''))
    sample_size = int(size / 1000 * len(ori_data))
    entity_sample = ori_data[:sample_size]
    with open(SAVE_FILE, 'w') as f:
        for item in tqdm(entity_sample):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# %%
dataset_fusion_labels = sorted(dataset_fusion_labels)
idx = 0
final_labels = {}
for label in ori_labels:
    final_labels[label] = {
        'idx': idx,
        'count': -1,
        'is_target': True
    }
    idx += 1
for label in dataset_fusion_labels:
    if label in count_dict:
        count = count_dict[label]
    else:
        count = 9999
    final_labels[label] = {
        'idx': idx,
        'count': count,
        'is_target': False
    }
    idx += 1
with open(SAVE_LABEL_FILE, 'w') as f:
    json.dump(final_labels, f, ensure_ascii=False)

# %%
