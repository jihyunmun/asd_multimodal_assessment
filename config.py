import torch
import random
import numpy as np

class Config:
    # For Huggingface ASR Models
    HUGGINGFACE_TOKEN = 'your token here'
    ASR_MODEL = 'your model here'

    # File paths
    TRAIN_CSV_PATH = 'train.csv'
    VALID_CSV_PATH = 'valid.csv'
    TEST_CSV_PATH = 'test.csv'

    TRAIN_EMB_DIR = 'emb/train'
    VALID_EMB_DIR = 'emb/valid'
    TEST_EMB_DIR = 'emb/test'

    CHECKPOINT_DIR = 'checkpoints'
    OUTPUT_JSON = 'output.json'

    MAX_TEXT_LENGTH = 128 # fit to the data
    TEXT_COLUMN_NAME = "asr_text"
    LABEL_COLUMN_NAME = "label"

    # Training settings
    SEEDS = [0, 1, 2, 3, 4]
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    PATIENCE = 20
    LR = 3e-5
    WEIGHT_DECAY = 1e-3
    CLIP_VALUE = 1.0
    NUM_WORKERS = 4

    # Dataset settings -> fit to your experimental settings
    DATASET_PARAMS = dict(
        use_speech = True,
        use_text = True,
        use_trillsson = True,
        use_w2v2 = True,
        preload_to_ram = True
    )

    # Device settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    """
    Set seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False