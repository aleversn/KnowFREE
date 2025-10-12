# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from main.trainers.knowfree_trainer import Trainer
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained(
    "chinese-bert-wwm-ext/")
config = BertConfig.from_pretrained(
    "chinese-bert-wwm-ext/")
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='chinese-bert-wwm-ext/',
                  data_name='weibo_250_fusion',
                  batch_size=4,
                  batch_size_eval=8,
                  task_name='CNNNER-weibo_250_fusion_Ro')

for i in trainer(num_epochs=120, other_lr=1e-3, weight_decay=0.01, remove_clashed=True, nested=False, eval_call_step=lambda x: x % 63 == 0):
    a = i

# %%
