import torch
from transformers import AutoTokenizer

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import DatasetType, get_dataset
from runner.predict import predict

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 模型
model = ProductClassifier().to(device)
model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')

# 数据
dataset = get_dataset(DatasetType.TRAIN)
class_label = dataset.features['label']

def predict_titile(text):
    return predict(text,model,tokenizer,device,class_label)
