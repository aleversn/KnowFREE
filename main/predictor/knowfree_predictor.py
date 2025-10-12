import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup
from main.models.knowfree import KnowFREE
from main.predictor.cnnner_predictor import Predictor
from main.loaders.span_loader import SpanNERDataset, SpanNERPadCollator
from main.utils.label_tokenizer import LabelTokenizer
from typing import List
from tqdm import tqdm


class KnowFREEPredictor(Predictor):
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
        self.config.num_target_labels = len(self.labelTokenizer) if self.num_target_labels is None else self.num_target_labels
        self.model = KnowFREE.from_pretrained(
            self.from_pretrained, config=self.config)
        