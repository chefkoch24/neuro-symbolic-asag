import torch
from training import *
from config import Config
import warnings
warnings.filterwarnings("ignore")

# define the fixed justification cue model here
TASK = 'token_classification'
#TASK = 'span_prediction'
#CHECKPOINT_PATH = 'logs/span_prediction_microsoft_mdeberta-v3-base/version_0/checkpoints/checkpoint-epoch=03-val_loss=1.35.ckpt'
CHECKPOINT_PATH = 'logs/token_classification_microsoft_mdeberta-v3-base_True/version_0/checkpoints/checkpoint-epoch=02-val_loss=0.40.ckpt'
#MODEL = 'distilbert-base-multilingual-cased'
MODEL = 'microsoft/mdeberta-v3-base'
CONTEXT = True
#CONTEXT =False
TRAIN_FILE = 'training_dataset.json'
DEV_FILE = 'dev_dataset.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

for mode in ['regression','classification']:
    for grading_model in ['summation']:#,'summation']:
            for th in [0.5, 0.6, 0.7, 0.8]:
                        config = Config(task=TASK,
                                        model=MODEL,
                                        train_file=TRAIN_FILE,
                                        dev_file=DEV_FILE,
                                        checkpoint_path=CHECKPOINT_PATH,
                                        context=CONTEXT,
                                        device=DEVICE,
                                        mode=mode,
                                        grading_model=grading_model,
                                        summation_th=th,
                                        batch_size=4,
                                        lr=0.01,
                                        matching='fuzzy',
                                        )
                        TrainingGrading(config).run_training()



