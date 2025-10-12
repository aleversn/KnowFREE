# 构造Dataloader
from typing import Any, Dict, List
from torch.utils.data import Dataset
from transformers import BertTokenizer
from main.utils.label_tokenizer import LabelTokenizer
from tqdm import tqdm
import torch
import json
import random


class SpanNERDataset(Dataset):

    def __init__(self, tokenizer: BertTokenizer, labelTokenizer: LabelTokenizer, filename: str, k_shot=None, shuffle=False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.labelTokenizer = labelTokenizer
        self.filename = filename
        self.num_labels = len(self.labelTokenizer)-1
        self.k_shot = k_shot
        self.data = self.load_jsonl()
        self.process_data = self.process_dataset()
        if self.k_shot is not None:
            self.process_data = self.k_shot_sample(self.process_data)
        self.shuffle_list = [i for i in range(len(self.process_data))]
        if shuffle:
            random.shuffle(self.shuffle_list)

    def load_jsonl(self):
        data = []
        assert self.filename.endswith(
            ".jsonl"), f"Invalid file format: {self.filename}"
        with open(self.filename, "r", encoding="utf-8") as f:
            from tqdm import tqdm
            bar = tqdm(f, desc=f"Loading {self.filename}")
            for line in bar:
                data.append(json.loads(line.strip()))
        return data

    def process_dataset(self):
        process_data = []
        for sample in tqdm(self.data, desc="converting data..."):
            process_data.append(SpanNERDataset.transform(self.tokenizer, self.labelTokenizer, sample, self.num_labels))
        return process_data
    
    @staticmethod
    def transform(tokenizer, labelTokenizer, sample: Dict[str, any], num_labels, model_len=512):
        convert_sample = {
            "input_ids": None,
            "bpe_len": None,
            "labels": None,
            "indexes": None
        }
        text, entities = sample["text"][:model_len - 2], sample["entities"]
        mask_ori = 0
        is_synthetic = 0
        # 用于测试混入dev和test数据集, 数据集会带有mask标识来告知模型是否mask原始标签
        if 'mask_ori' in sample:
            mask_ori = 1 if sample['mask_ori'] else 0
        if 'synthetic' in sample:
            is_synthetic = 1 if sample['synthetic'] else 0
        pieces = list(tokenizer.tokenize(word) for word in text)
        pieces = list(tokenizer.unk_token if len(
            piece) == 0 else piece for piece in pieces)
        flat_tokens = [i for piece in pieces for i in piece]
        length = len(text)
        bert_length = len(flat_tokens) + 2
        input_ids = torch.tensor([tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
            flat_tokens) + [tokenizer.sep_token_id], dtype=torch.long)
        labels = torch.zeros(
            (length, length, num_labels), dtype=torch.long)
        for entity in entities:
            if entity["start"] > model_len - 3 or entity["end"] > model_len - 3:
                continue
            start, end, label = entity["start"], entity["end"] - \
                1, entity["entity"]
            label_id = labelTokenizer.convert_tokens_to_ids(label)
            labels[start, end, label_id-1] = 1
            # 原论文中是计算的上下三角形，但是实际的话是只使用上边的三角形效果好
            # labels[end, start, label_id-1] = 1
        indexes = torch.zeros(bert_length, dtype=torch.long)
        offset = 0
        for i, piece in enumerate(pieces):
            indexes[offset+1: offset+len(piece) + 1] = i + 1
            offset += len(piece)
        convert_sample["input_ids"] = input_ids
        convert_sample["bpe_len"] = torch.tensor(bert_length, dtype=torch.long)
        convert_sample["labels"] = labels
        convert_sample["indexes"] = indexes
        convert_sample['mask_ori'] = torch.tensor(mask_ori)
        convert_sample['is_synthetic'] = torch.tensor(is_synthetic, dtype=torch.long)
        return convert_sample
    
    def k_shot_sample(self, data):
        ori_data = []
        syn_data = []
        for item in data:
            if item['is_synthetic'].tolist() == 1:
                syn_data.append(item)
            else:
                ori_data.append(item)
        if len(syn_data) >= self.k_shot * 3:
            syn_data = random.sample(syn_data, self.k_shot * 3)
        return random.sample(ori_data, self.k_shot) + syn_data

    def __getitem__(self, index) -> Any:
        index = self.shuffle_list[index]
        return self.process_data[index]

    def __len__(self) -> int:
        return len(self.process_data)


class SpanNERPadCollator:

    def __init__(self):
        pass

    def pad_1d(self, x: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
        max_length = max(i.size(0) for i in x)
        paddings = torch.full((len(x), max_length) +
                              x[0].size()[2:], padding_value, dtype=x[0].dtype)
        for i in range(len(x)):
            paddings[i, :x[i].size(0)] = x[i]
        return paddings

    def pad_2d(self, x: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
        """
        对序列进行二维补全
        """
        max_rows = max(i.size(0) for i in x)
        max_cols = max(i.size(1) for i in x)
        paddings = torch.full((len(x), max_rows, max_cols) +
                              x[0].size()[2:], padding_value, dtype=x[0].dtype)
        # print(paddings.size(),x[0].size())
        for i in range(len(x)):
            paddings[i, :x[i].size(0), :x[i].size(1)] = x[i]
        return paddings

    def __call__(self, samples: List[Any]):
        # for i in samples:
        #     print(i['labels'].size())
        convert_example = {
            "input_ids": self.pad_1d(list(i["input_ids"] for i in samples), 0),
            "bpe_len": torch.stack(list(i["bpe_len"] for i in samples)),
            "labels": self.pad_2d(list(i["labels"] for i in samples), 0),
            "indexes": self.pad_1d(list(i["indexes"] for i in samples), 0),
            "mask_ori": torch.stack(list(i["mask_ori"] for i in samples)),
            'is_synthetic': torch.stack(list(i['is_synthetic'] for i in samples))
        }

        return convert_example
