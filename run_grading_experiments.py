import torch
from training import *
from config import Config

# define the fixed justification cue model here
#TASK = 'token_classification'
TASK = 'span_prediction'
#CHECKPOINT_PATH = 'logs/token_classification_distilbert-base-multilingual-cased_True/version_0/checkpoints/checkpoint-epoch=02-val_loss=0.63.ckpt'
CHECKPOINT_PATH = 'logs/span_prediction_distilbert-base-multilingual-cased/version_0/checkpoints/checkpoint-epoch=01-val_loss=1.52.ckpt'
MODEL = 'distilbert-base-multilingual-cased'
#CONTEXT = True
CONTEXT =False
TRAIN_FILE = 'training_dataset.json'
DEV_FILE = 'dev_dataset.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

for mode in ['classification', 'regression']:
    for grading_model in ['summation', 'decision_tree']:
        config = Config(task=TASK,
                        model=MODEL,
                        train_file=TRAIN_FILE,
                        dev_file=DEV_FILE,
                        checkpoint_path=CHECKPOINT_PATH,
                        context=CONTEXT,
                        device=DEVICE,
                        mode=mode,
                        grading_model=grading_model,
                        summation_th=0.9,
                        )
        TrainingGrading(config).run_training()



