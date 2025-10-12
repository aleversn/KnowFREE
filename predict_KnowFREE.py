# %%
from main.predictor.knowfree_predictor import KnowFREEPredictor
from transformers import BertTokenizer, BertConfig

MODEL_PATH = "cnnner_best/"
LABEL_FILE = '1000/labels_fusion.json'
tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext/')
config = BertConfig.from_pretrained(MODEL_PATH)
pred = KnowFREEPredictor(tokenizer=tokenizer, config=config, from_pretrained=MODEL_PATH, label_file=LABEL_FILE, batch_size=4)

for entities in pred(['叶赟葆：全球时尚财运滚滚而来钱', '我要去我要去花心花心花心耶分手大师贵仔邓超四大名捕围观话筒转发邓超贴吧微博号外话筒望周知。邓超四大名捕']):
    print(entities)
