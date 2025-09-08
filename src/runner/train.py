import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader, DatasetType


class EarlyStoping:
    def __init__(self,patience=3):
        self.best_loss = float('inf')
        self.counter = 0 # 连续超过best_loss的次数
        self.patience = patience

    def should_stop(self,avg_loss,model,path):
        if avg_loss < self.best_loss:
            self.counter = 0
            self.best_loss = avg_loss
            torch.save(model.state_dict(),path)
            print("保存最优模型")
            return False
        else:
            self.counter +=1
            if self.counter >= self.patience:
                return True
            else:
                return False




def run_one_epoch(model, dataloader, loss_function, device,optimizer=None ,is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader,desc=("训练" if is_train else "验证")):
            input_ids = batch['input_ids'].to(device) # [batch_size, seq_len]
            attention_mask = batch['attention_mask'].to(device) # [batch_size, seq_len]
            label = batch['label'].to(device) # [batch_size]

            # 前向传播
            outputs = model(input_ids,attention_mask) # [batch_size, num_classes]
            loss = loss_function(outputs,label)
            if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_loss+=loss.item()
    return total_loss/len(dataloader)

def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    train_dataloader = get_dataloader()
    valid_dataloader = get_dataloader(DatasetType.VALID)
    # 模型
    model = ProductClassifier(freeze_bert=config.FREEZE_BERT).to(device)
    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 日志
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 早停策略
    early_stopping = EarlyStoping()


    for epoch in range(1,config.EPOCHS+1):
        print(f"========== epoch:{epoch}==========")
        # 训练一轮
        train_avg_loss = run_one_epoch(model, train_dataloader, loss_function, device,optimizer,is_train=True )
        # 验证一轮
        valid_avg_loss = run_one_epoch(model, valid_dataloader, loss_function, device, optimizer,is_train=False)

        print(f"TrainLoss:{train_avg_loss}")
        print(f"ValidLoss:{valid_avg_loss}")
        writer.add_scalar('Loss/train',train_avg_loss,epoch)
        writer.add_scalar('Loss/valid',valid_avg_loss,epoch)

        if early_stopping.should_stop(valid_avg_loss,model,path=config.MODELS_DIR / 'best.pt'):
            print("早停策略触发，训练提前结束！")
            break

    writer.close()