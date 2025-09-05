from torch import nn
from transformers import AutoModel

from configuration import config


class ProductClassifier(nn.Module):

    def __init__(self,freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')
        self.linear = nn.Linear(in_features=self.bert.config.hidden_size, out_features=config.NUM_CLASSES)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad=False

    def forward(self,input_ids,attention_mask):
        output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden = output.last_hidden_state # [batch_size, seq_len, hidden_size]
        cls_output = last_hidden[:,0,:] # [batch_size, hidden_size]
        return self.linear(cls_output) # [batch_size, num_classes]