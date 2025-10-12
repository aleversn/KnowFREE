# %%
import os
from argparse import ArgumentParser

# if you want to run through jupyter, please set it as false.
import sys
sys.path.append("../")
cmd_args = False

parser = ArgumentParser()
parser.add_argument('--file_dir', default='<YOUR_DATASET_DIR>', help='file name')
parser.add_argument('--file_name', default='conll_2003', help='file name of the dataset, you should make sure it contains `train_1000.jsonl` file')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

SOURCE_FILE = os.path.join(args.file_dir, args.file_name, 'train_1000.jsonl')
SAVE_DIR = os.path.dirname(SOURCE_FILE) + f'_{args.save_type_name}_DA'

suffix_list = ['', '_synthetic']
for suffix in suffix_list:
    ori_basename = os.path.basename(SOURCE_FILE).split('.')[0]
    basename = ori_basename + suffix + '.jsonl'

    split_list = [250, 500, 1000]

    with open(os.path.join(SAVE_DIR, f'entity_{basename}')) as f:
        ori_entity_file = f.readlines()

    with open(os.path.join(SAVE_DIR, f'pos_{basename}')) as f:
        ori_pos_file = f.readlines()

    for size in split_list:
        sample_size = int(size / 1000 * len(ori_entity_file))
        entity_sample = ori_entity_file[:sample_size]
        pos_sample = ori_pos_file[:sample_size]
        save_entity_name = f'entity_{basename}'
        save_pos_name = f'pos_{basename}'
        save_entity_name = save_entity_name.replace('1000', str(size))
        save_pos_name = save_pos_name.replace('1000', str(size))
        save_dir = os.path.join(SAVE_DIR, str(size))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, save_entity_name), mode='w+') as f:
            for item in entity_sample:
                f.write(item)
        with open(os.path.join(save_dir, save_pos_name), mode='w+') as f:
            for item in pos_sample:
                f.write(item)

basename = os.path.basename(SOURCE_FILE).split('.')[0] + '_synthetic.jsonl'
with open(os.path.join(SAVE_DIR, basename)) as f:
    ori_syn_file = f.readlines()

for size in split_list:
    sample_size = int(size / 1000 * len(ori_entity_file))
    syn_sample = ori_syn_file[:sample_size]
    save_syn_name = basename.replace('1000', str(size))
    save_dir = os.path.join(SAVE_DIR, str(size))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, save_syn_name), mode='w+') as f:
        for item in syn_sample:
            f.write(item)

# %%
