import sys

from preprocess.dataset import get_dataloader, DatasetType
from preprocess.process import process_data
from runner.predict import run_predict
from runner.train import train

if __name__ == '__main__':
    print(sys.path)
    # process_data()
    # train_dataloader = get_dataloader(data_type=DatasetType.TRAIN)
    # test_dataloader = get_dataloader(data_type=DatasetType.TEST)
    # print(len(train_dataloader))
    # print(len(test_dataloader))

    # train.py
    # train()
    # predict.py
    run_predict()
