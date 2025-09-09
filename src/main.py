import sys
from argparse import ArgumentParser

import uvicorn

from preprocess.dataset import get_dataloader, DatasetType
from preprocess.process import process_data




if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('command',choices=['train','predict','evaluate','preprocess','server'])
    args = arg_parser.parse_args()
    command = args.command
    if command == 'train':
        from runner.train import train
        train()
    elif command == 'predict':
        from runner.predict import run_predict
        run_predict()
    elif command == 'evaluate':
        from runner.evaluate import run_evaluate
        run_evaluate()
    elif command == 'preprocess':
        process_data()
    elif command == 'server':
        from web.app import run_app
        run_app()