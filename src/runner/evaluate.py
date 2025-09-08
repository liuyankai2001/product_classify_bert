import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader, DatasetType
from runner.predict import predict_batch


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predications = []
    for batch in tqdm(dataloader,desc='评估'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].tolist()
        predict_result = predict_batch(input_ids, attention_mask, model)
        all_labels.extend(label)
        all_predications.extend(predict_result)
    accuracy = accuracy_score(all_labels,all_predications)
    precision = precision_score(all_labels,all_predications,average='macro')
    recall = recall_score(all_labels,all_predications,average='macro')
    f1 = f1_score(all_labels,all_predications,average='macro')
    return {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}



def run_evaluate():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型
    model = ProductClassifier().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    # tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')

    # 数据
    dataloader = get_dataloader(DatasetType.TEST)

    result = evaluate_model(model,dataloader,device)
    print("========== 评估结果 ==========")
    print(f"accuracy:{result['accuracy']:.4f}")
    print(f"precision:{result['precision']:.4f}")
    print(f"recall:{result['recall']:.4f}")
    print(f"f1:{result['f1']:.4f}")
    print("=============================")