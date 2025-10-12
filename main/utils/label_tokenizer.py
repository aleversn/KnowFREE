import os
import json
from typing import List, Union
from tqdm import tqdm

class LabelTokenizer():
    
    def __init__(self, filename: str=None, k=10, b=0, is_absolute_weight: bool=False) -> None:
        self.idx2label = []
        self.label2idx = {}
        self.label_weights = []
        self.ori_label_count = 0
        if filename is not None:
            if not os.path.basename(filename).endswith('.json'):
                with open(filename, 'r') as f:
                    for idx, line in tqdm(enumerate(f), desc='Loading labels'):
                        self.idx2label.append(line.strip())
                        self.label2idx[line.strip()] = idx
                        self.label_weights.append(1.0)
                        self.ori_label_count += 1
            else:
                if os.path.basename(filename).endswith('.json'):
                    with open(filename, 'r') as f:
                        label_data = json.load(f)
                elif os.path.basename(filename).endswith('.jsonl'):
                    with open(filename, 'r') as f:
                        label_data = f.readlines()
                    label_data = [json.loads(line) for line in label_data]
                else:
                    raise ValueError('Unsupported file format')
                idx = 0
                for key in label_data.keys():
                    self.idx2label.append(key.strip())
                    self.label2idx[key.strip()] = idx
                    if 'is_target' in label_data[key] and label_data[key]['is_target']:
                        self.label_weights.append(1.0)
                        self.ori_label_count += 1
                    else:
                        if is_absolute_weight:
                            self.label_weights.append(0.13)
                        elif 'weight' in label_data[key]:
                            self.label_weights.append(float(label_data[key]['weight']))
                        else:
                            weight = (k / float(label_data[key]['count']) + b)
                            self.label_weights.append('count_weight:{:.4f}'.format(weight))
                    idx += 1
                max_threshold = (self.ori_label_count - 1) / (len(label_data.keys()) - 1)
                for i in range(len(self.label_weights)):
                    if isinstance(self.label_weights[i], str):
                        weight = float(self.label_weights[i].split(':')[-1])
                        if weight > max_threshold:
                            self.label_weights[i] = max_threshold
                        else:
                            self.label_weights[i] = weight
                

    def load(self, labels: List[str], sort=True):
        if sort:
            # 排序确保顺序
            labels = list(labels)
            labels.sort()
        for label in labels:
            self.idx2label.append(label)
        for idx, label in enumerate(self.idx2label):
            self.label2idx[label] = idx
        return self
    
    def remove(self, index: int):
        label = self.idx2label[index]
        self.idx2label.remove(label)
        self.label2idx = {}
        for idx, label in enumerate(self.idx2label):
            self.label2idx[label] = idx
        return self
    
    def add(self, item, index=-1):
        if index!=-1:
            self.idx2label.insert(index, item)
        else:
            self.idx2label.append(item)
        for idx, label in enumerate(self.idx2label):
            self.label2idx[label] = idx
        return self
    
    def convert_tokens_to_ids(self, label: Union[str, List[str]]) -> int:
        if isinstance(label, list):
            return [self.label2idx[l] for l in label]
        return self.label2idx[label]
    
    def convert_ids_to_tokens(self, idx: Union[int, List[int]]) -> int:
        if isinstance(idx, list):
            return [self.idx2label[i] for i in idx]
        return self.idx2label[idx]
    
    def __len__(self):
        return len(self.idx2label)