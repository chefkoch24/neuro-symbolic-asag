import torch
from pytorch_lightning.callbacks import EarlyStopping

from training import *
from config import Config
import warnings
warnings.filterwarnings("ignore")

# define the fixed justification cue model here
MODEL = 'microsoft/mdeberta-v3-base'
TRAIN_FILE = 'training_dataset.json'
DEV_FILE = 'dev_dataset.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EarlyStopping(
            monitor='val_loss',
            min_delta=0.0,
            patience=3,
            mode='min'
        )

for task in ['token_classification', 'span_prediction']:
    if task == 'token_classification':
        ckpt = 'logs/token_classification_microsoft_mdeberta-v3-base_True/version_0/checkpoints/checkpoint-epoch=03-val_loss=0.40.ckpt'
    else:
        ckpt = 'logs/span_prediction_microsoft_mdeberta-v3-base/version_0/checkpoints/checkpoint-epoch=03-val_loss=1.35.ckpt'
    for mode in ['regression','classification']:
        for grading_model in ['decision_tree', 'summation']:
                        config = Config(task=task,
                                        model=MODEL,
                                        train_file=TRAIN_FILE,
                                        dev_file=DEV_FILE,
                                        checkpoint_path=ckpt,
                                        context=True,
                                        device=DEVICE,
                                        mode=mode,
                                        grading_model=grading_model,
                                        summation_th=0.9, # TODO:
                                        batch_size=4,
                                        lr=0.001,
                                        matching='exact',
                                        is_fixed_learner=False,
                                        )
                        TrainingGrading(config).run_training()



