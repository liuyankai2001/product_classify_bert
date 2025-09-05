from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
import pandas as pd
from configuration import config


def process_data():
    print("开始处理数据")
    dataset_dict = load_dataset('csv',data_files={
        'train':str(config.RAW_DATA_DIR / 'train.txt'),
        'valid':str(config.RAW_DATA_DIR / 'valid.txt'),
        'test':str(config.RAW_DATA_DIR / 'test.txt'),
    },delimiter='\t')
    # 过滤数据
    dataset_dict = dataset_dict.filter(lambda x:x['label'] is not None and x['text_a'] is not None)

    # 构建数据集
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_DIR / 'bert-base-chinese')
    # 统计标题长度
    # df = dataset_dict['train'].to_pandas()
    # print(df['text_a'].apply(lambda x: len(tokenizer.tokenize(x))).max())


    def tokenize(batch):
        tokenized = tokenizer(batch['text_a'], padding='max_length', truncation=True, max_length = config.SEQ_LEN)
        return {'input_ids':tokenized['input_ids'],
                'attention_mask':tokenized['attention_mask']}
    # 处理text_a
    dataset_dict = dataset_dict.map(tokenize,batched=True,remove_columns=['text_a'])
    # 处理label
    all_labels = dataset_dict['train'].unique('label')
    # print(f"{len(all_labels) = }")
    class_label = ClassLabel(names=all_labels)
    dataset_dict = dataset_dict.cast_column('label',class_label)
    # 保存数据集
    dataset_dict['train'].save_to_disk(config.PROCESS_DATA_DIR / 'train')
    dataset_dict['test'].save_to_disk(config.PROCESS_DATA_DIR / 'test')
    dataset_dict['valid'].save_to_disk(config.PROCESS_DATA_DIR / 'valid')

    print("处理数据完成")


