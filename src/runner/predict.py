import torch
from transformers import AutoTokenizer

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader, DatasetType, get_dataset


def predict_batch(input_ids, attention_mask, model):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids,attention_mask) # [batch_size, num_classes]
        predicts = torch.argmax(outputs,dim=1) # [batch_size]
        return predicts.tolist()

def predict(text, model, tokenizer, device,class_label):
    # 处理数据
    encoded = tokenizer([text], padding='max_length', truncation=True, max_length=config.SEQ_LEN,return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    batch_result = predict_batch(input_ids,attention_mask,model)
    result = batch_result[0]
    return class_label.int2str(result)



def run_predict():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型
    model = ProductClassifier().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')

    # 数据
    dataset = get_dataset(DatasetType.TRAIN)
    class_label = dataset.features['label']

    print("开始预测...")
    print("请输入商品标题（q或者quit退出）")
    while True:
        text = input("> ")
        if text in ['q','quit']:
            break
        if not text:
            continue
        class_name = predict(text,model,tokenizer,device,class_label)
        print("所属类别：",class_name)