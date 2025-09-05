import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader,desc="训练"):
        input_ids = batch['input_ids'].to(device) # [batch_size, seq_len]
        attention_mask = batch['attention_mask'].to(device) # [batch_size, seq_len]
        label = batch['label'].to(device) # [batch_size]
        optimizer.zero_grad()
        # 前向传播
        outputs = model(input_ids,attention_mask) # [batch_size, num_classes]
        loss = loss_function(outputs,label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(dataloader)

def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    dataloader = get_dataloader()
    # 模型
    model = ProductClassifier(freeze_bert=config.FREEZE_BERT).to(device)
    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 日志
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))
    best_loss = float('inf')
    for epoch in range(1,config.EPOCHS+1):
        print(f"========== epoch:{epoch}==========")
        avg_loss = train_one_epoch(model,dataloader,loss_function,optimizer,device)
        print(f"Loss:{avg_loss}")
        writer.add_scalar('Loss/train',avg_loss,epoch)
        if avg_loss < best_loss:
            best_loss=avg_loss
            torch.save(model.state_dict(),config.MODELS_DIR / 'best.pt')
            print("保存模型")
        else:
            print("模型无需保存")
    writer.close()