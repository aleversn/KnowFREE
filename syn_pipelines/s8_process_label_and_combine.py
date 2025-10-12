# %%
from s5_process_label import *

for size in split_list:
    FUSION_FILE = os.path.join(DA_DIR, str(
        size), f'train_{str(size)}_fusion.jsonl')
    FUSION_SYN_FILE = os.path.join(DA_DIR, str(
        size), f'train_{str(size)}_synthetic_fusion.jsonl')
    with open(FUSION_FILE, 'r', encoding='utf-8') as f:
        fusion_data = f.readlines()
    with open(FUSION_SYN_FILE, encoding='utf-8') as f:
        fusion_syn_data = f.readlines()
    fusion_syn_data.extend(fusion_data)
    with open(FUSION_SYN_FILE, 'w', encoding='utf-8') as f:
        for item in fusion_syn_data:
            f.write(item)

# %%
