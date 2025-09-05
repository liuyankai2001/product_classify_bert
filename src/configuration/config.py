from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESS_DATA_DIR = ROOT_DIR / 'data' / 'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'

PRETRAINED_DIR = ROOT_DIR / 'pretrained'
SEQ_LEN = 50
NUM_CLASSES = 30

BATCH_SIZE = 32
FREEZE_BERT = False
LEARNING_RATE = 1e-5
EPOCHS = 10
