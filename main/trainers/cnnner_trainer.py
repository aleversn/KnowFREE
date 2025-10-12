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
from main.analysis.anlysis import Analysis
from main.analysis.metrics import SpanNERMetrics
from main.utils.label_tokenizer import LabelTokenizer
from torch.utils.data import DataLoader
from typing import List
from prettytable import PrettyTable
from tqdm import tqdm


class Trainer():
    def __init__(self, tokenizer,
                 from_pretrained=None,
                 data_name='default',
                 data_present_path='./datasets/present.json',
                 train_file=None,
                 eval_file=None,
                 test_file=None,
                 label_file=None,
                 batch_size=8,
                 batch_size_eval=8,
                 n_head: int = 4,
                 cnn_dim: int = 200,
                 span_threshold: float = 0.5,
                 size_embed_dim: int = 25,
                 biaffine_size: int = 200,
                 logit_drop: int = 0,
                 kernel_size: int = 3,
                 cnn_depth: int = 3,
                 warmup_steps: int = 190,
                 eval_mode='test',
                 task_name='CNNNER',
                 **args):

        self.tokenizer = tokenizer
        self.from_pretrained = from_pretrained
        self.data_name = data_name
        self.data_present_path = data_present_path
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file
        self.label_file = label_file
        self.task_name = task_name
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval

        self.n_head = n_head
        self.cnn_dim = cnn_dim
        self.span_threshold = span_threshold
        self.size_embed_dim = size_embed_dim
        self.biaffine_size = biaffine_size
        self.logit_drop = logit_drop
        self.kernel_size = kernel_size
        self.cnn_depth = cnn_depth

        self.warmup_steps = warmup_steps

        self.eval_mode = eval_mode
        self.pred_gold = []

        self.dataloader_init()
        self.model_init()
        self.analysis = Analysis()
        self.metric_fn = SpanNERMetrics(self.labelTokenizer)

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

        self.pretrained_parameters = []
        self.delay_pretrained_parameters = []
        self.other_parameters = []
        self.delay_other_parameters = []
        for name, param in self.model.named_parameters():
            name = name.lower()
            if param.requires_grad is False:
                continue
            if "bert" in name:
                if "bias" in name or "norm" in name:
                    self.delay_pretrained_parameters.append(param)
                else:
                    self.pretrained_parameters.append(param)
            else:
                if "bias" in name or "norm" in name:
                    self.delay_other_parameters.append(param)
                else:
                    self.other_parameters.append(param)

    def dataloader_init(self):
        if self.data_present_path is None:
            self.data_path = {
                'train': self.train_file,
                'dev': self.eval_file,
                'test': self.test_file,
                'labels': self.label_file,
            }
        else:
            self.data_path = self.get_data_present(
                self.data_present_path)[self.data_name]

        self.labelTokenizer = LabelTokenizer(self.data_path['labels'])

        collate_fn = SpanNERPadCollator()

        self.train_set = SpanNERDataset(
            self.tokenizer, self.labelTokenizer, self.data_path['train'], shuffle=True)
        self.eval_set = SpanNERDataset(
            self.tokenizer, self.labelTokenizer, self.data_path['dev'], shuffle=False)
        if 'test' in self.data_path and self.data_path['test'] is not None:
            self.test_set = SpanNERDataset(
                self.tokenizer, self.labelTokenizer, self.data_path['test'], shuffle=False)

        self.train_loader = DataLoader(
            self.train_set, self.batch_size, collate_fn=collate_fn)
        self.eval_loader = DataLoader(
            self.eval_set, self.batch_size_eval, collate_fn=collate_fn) if self.eval_mode == 'dev' else DataLoader(self.test_set, self.batch_size_eval, collate_fn=collate_fn)

    def get_data_present(self, present_path):
        if not os.path.exists(present_path):
            return {}
        with open(present_path, encoding='utf-8') as f:
            present_json = f.read()
        data_present = json.loads(present_json)
        return data_present

    def model_to_device(self, gpu=[0]):
        self.num_gpus = len(gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)

    def __call__(self, resume_step=None, num_epochs=30, pretrained_lr=2e-5, other_lr=2e-3, weight_decay=0.01, remove_clashed=False, nested=False, gpu=[0], eval_call_step=None, save_per_call=False):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs,
                          pretrained_lr=pretrained_lr,
                          other_lr=other_lr,
                          weight_decay=weight_decay,
                          remove_clashed=remove_clashed,
                          nested=nested,
                          gpu=gpu,
                          eval_call_step=eval_call_step,
                          save_per_call=save_per_call
                          )

    def train(self, resume_step=None, num_epochs=30, pretrained_lr=2e-5, other_lr=2e-3, weight_decay=0.01, remove_clashed=False, nested=False, gpu=[0], eval_call_step=None, save_per_call=False, save_per_epoch=False):
        self.model_to_device(gpu=gpu)

        optimizer = optim.AdamW([
            {"params": self.pretrained_parameters, "lr": pretrained_lr},
            {"params": self.delay_pretrained_parameters,
                "lr": pretrained_lr, "weight_decay": weight_decay},
            {"params": self.other_parameters, "lr": other_lr},
            {"params": self.delay_other_parameters,
                "lr": other_lr, "weight_decay": weight_decay}
        ])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.warmup_steps, len(self.train_loader) * num_epochs)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        best_eval_score = -1
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0

            train_iter = tqdm(self.train_loader)
            self.model.train()

            for it in train_iter:
                for key in it.keys():
                    if isinstance(it[key], torch.Tensor):
                        it[key] = self.cuda(it[key])

                loss = self.model(**it)[0]
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                train_loss += loss.data.item()
                train_count += 1
                train_step += 1

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count)

                if (eval_call_step is None and train_step % 125 == 0) or (eval_call_step is not None and eval_call_step(train_step)):
                    report = self.eval(
                        train_step, remove_clashed=remove_clashed, nested=nested)
                    if report['f1'] > best_eval_score:
                        best_eval_score = report['f1']
                        self.save_model('best')
                        self.analysis.save_list(uid=current_uid if self.task_name is None else self.task_name, filename='pred_gold_best.jsonl', content=self.pred_gold)
                    elif save_per_call:
                        self.save_model(train_step, '_step')
                        self.analysis.save_list(uid=current_uid if self.task_name is None else self.task_name, filename=f'pred_gold_{train_step}.jsonl', content=self.pred_gold)
                    self.analysis.save_all_records(
                        uid=current_uid if self.task_name is None else self.task_name)
                    yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, 'current_best')
                    self.model.train()
            
            if save_per_epoch:
                model_uid = self.save_model(epoch + 1)
            else:
                model_uid = 'not_saving'

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count
            })

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    def save_model(self, current_step=0, prefix=''):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists(f'./save_model/{dir}'):
            os.makedirs(f'./save_model/{dir}')
        model_self = self.model.module if hasattr(
            self.model, 'module') else self.model
        model_self.save_pretrained(
            f'./save_model/{dir}/cnnner{prefix}_{current_step}', safe_serialization=False)
        self.tokenizer.save_pretrained(f'./save_model/{dir}/cnnner{prefix}_{current_step}')
        self.analysis.append_model_record(current_step)
        return current_step
    
    def update_pred_golds(self, preds, golds):
        for preds, golds in zip(preds, golds):
            item = {
                'preds': [],
                'golds': []
            }
            preds = list(preds)
            for pred in preds:
                item['preds'].append({
                    'start': int(pred[0]),
                    'end': int(pred[1]),
                    'label': int(pred[2])
                })
            for gold in golds:
                item['golds'].append({
                    'start': int(gold[0]),
                    'end': int(gold[1]),
                    'label': int(gold[2])
                })
            self.pred_gold.append(item)
        

    def eval(self, epoch, gpu=[0], is_eval=False, remove_clashed=False, nested=False):
        if is_eval:
            self.model_to_device(gpu=gpu)
        self.metric_fn.reset()
        self.pred_gold = []
        with torch.no_grad():
            eval_count = 0
            eval_loss = 0

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            for it in eval_iter:
                for key in it.keys():
                    if isinstance(it[key], torch.Tensor):
                        it[key] = self.cuda(it[key])

                loss, scores = self.model(**it)
                loss = loss.mean()

                eval_loss += loss.data.item()
                eval_count += 1

                model_self = self.model.module if hasattr(
                    self.model, 'module') else self.model
                entities: List[set] = model_self.decode_logits(
                    scores, it["indexes"], remove_clashed, nested)
                gold_entities: List[set] = model_self.decode_labels(
                    it["labels"], it["indexes"])
                self.update_pred_golds(entities, gold_entities)
                self.metric_fn.add(entities, gold_entities)

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count)
            eval_report = self.metric_fn.compute()
            table = PrettyTable()

            table.field_names = ["Metric"] + [metric for metric in eval_report.keys()][:8]
            table.add_row(["Scores"] + [round(score, 4) for score in eval_report.values()][:8])
            print(table)

            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count,
                **eval_report
            })

        return eval_report

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
