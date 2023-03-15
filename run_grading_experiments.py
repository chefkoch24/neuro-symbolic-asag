import torch
from training import *
from config import Config

TASK = 'token_classification'
CHECKPOINT_PATH = 'logs/token_classification_distilbert-base-multilingual-cased_bs-8_aggr-lfs_sum_context-True/version_0/checkpoints/checkpoint-epoch=00-val_loss=0.67.ckpt'
MODEL = 'distilbert-base-multilingual-cased'
CONTEXT = True
TRAIN_FILE = 'training_dataset.json'
DEV_FILE = 'dev_dataset.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

for mode in ['classification', 'regression']:
    config = Config(task=TASK,
                    model=MODEL,
                    train_file=TRAIN_FILE,
                    dev_file=DEV_FILE,
                    checkpoint_path=CHECKPOINT_PATH,
                    context=CONTEXT,
                    device=DEVICE,
                    mode=mode,
                    )
    for grading_model in ['decision_tree']:
        config.GRADING_MODEL = grading_model
        TrainingGrading(config).run_training()



