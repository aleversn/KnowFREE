import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup
from main.models.cnnner import CNNNerv1
from main.loaders.span_loader import SpanNERDataset, SpanNERPadCollator
from main.utils.label_tokenizer import LabelTokenizer
from typing import List
from tqdm import tqdm


class Predictor():
    def __init__(self, tokenizer,
                 from_pretrained=None,
                 label_file=None,
                 batch_size=8,
                 n_head: int = 4,
                 cnn_dim: int = 200,
                 span_threshold: float = 0.5,
                 size_embed_dim: int = 25,
                 biaffine_size: int = 200,
                 logit_drop: int = 0,
                 kernel_size: int = 3,
                 cnn_depth: int = 3,
                 **args):

        self.tokenizer = tokenizer
        self.from_pretrained = from_pretrained
        self.label_file = label_file
        self.batch_size = batch_size

        self.n_head = n_head
        self.cnn_dim = cnn_dim
        self.span_threshold = span_threshold
        self.size_embed_dim = size_embed_dim
        self.biaffine_size = biaffine_size
        self.logit_drop = logit_drop
        self.kernel_size = kernel_size
        self.cnn_depth = cnn_depth
        
        self.model_loaded = False

        self.load_labels()
        self.model_init()

        self.collate_fn = SpanNERPadCollator()

    def model_init(self):
        self.config = AutoConfig.from_pretrained(
            self.from_pretrained)
        self.config.num_labels = len(self.labelTokenizer)
        self.config.n_head = self.n_head
        self.config.cnn_dim = self.cnn_dim
        self.config.span_threshold = self.span_threshold
        self.config.size_embed_dim = self.size_embed_dim
        self.config.biaffine_size = self.biaffine_size
        self.config.logit_drop = self.logit_drop
        self.config.kernel_size = self.kernel_size
        self.config.cnn_depth = self.cnn_depth
        self.model = CNNNerv1.from_pretrained(
            self.from_pretrained, config=self.config)

    def load_labels(self):
        self.labelTokenizer = LabelTokenizer(self.label_file)
        self.num_labels = len(self.labelTokenizer) - 1
        self.num_target_labels = self.labelTokenizer.ori_label_count - 1

    def model_to_device(self, gpu=[0]):
        if self.model_loaded:
            return
        self.num_gpus = len(gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)
        self.model_loaded = True

    def __call__(self, inputs, gpu=[0], remove_clashed=False, nested=False):
        return self.pred(inputs, gpu=gpu, remove_clashed=remove_clashed, nested=nested)

    def pred(self, inputs, gpu=[0], remove_clashed=False, nested=False, skip_label_idxs=[]):
        self.model_to_device(gpu=gpu)
        skip_label_idxs = set(skip_label_idxs)
        model_self = self.model.module if hasattr(
            self.model, 'module') else self.model
        if isinstance(inputs, str):
            inputs = [inputs]
        num_batches = len(inputs) // self.batch_size + 1 if len(
            inputs) % self.batch_size != 0 else len(inputs) // self.batch_size
        with torch.no_grad():
            self.model.eval()
            for i in tqdm(range(num_batches)):
                batch_inputs = inputs[i*self.batch_size:(i+1)*self.batch_size]
                samples = []
                for item in batch_inputs:
                    if type(item) == str:
                        samples.append({
                            'text': list(item),
                            'entities': []
                        })
                    elif type(item) == list:
                        samples.append({
                            'text': item,
                            'entities': []
                        })
                    else:
                        raise 'input type error'
                transform_samples = []
                for sample in samples:
                    tr = SpanNERDataset.transform(
                        self.tokenizer, self.labelTokenizer, sample, self.num_labels)
                    transform_samples.append(tr)
                batch_transform_samples = self.collate_fn(transform_samples)
                for key in batch_transform_samples.keys():
                    batch_transform_samples[key] = self.cuda(
                        batch_transform_samples[key])
                loss, scores = self.model(**batch_transform_samples)
                entities: List[set] = model_self.decode_logits(
                    scores, batch_transform_samples["indexes"], remove_clashed, nested)
                result = []
                for idx, item_entities in enumerate(entities):
                    item_result = []
                    for entity in item_entities:
                        start, end, label_idx = int(entity[0]), int(entity[1]) + 1, int(entity[2])
                        if label_idx in skip_label_idxs:
                            continue
                        item_result.append({
                            'start': start,
                            'end': end,
                            'entity': self.labelTokenizer.convert_ids_to_tokens(label_idx),
                            'text': list(batch_inputs[idx][start:end])
                        })
                    result.append(item_result)
                yield result

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX
