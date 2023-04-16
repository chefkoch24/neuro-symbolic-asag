from training import *
from config import Config
import warnings
warnings.filterwarnings("ignore")

# define the fixed justification cue model here
MODEL = 'microsoft/mdeberta-v3-base'
TRAIN_FILE = 'training_dataset.json'
DEV_FILE = 'dev_dataset.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


for task in ['span_prediction', 'token_classification']:
    if task == 'token_classification':
        ckpt = 'logs/token_classification_microsoft_mdeberta-v3-base_True/version_0/checkpoints/checkpoint-epoch=05-val_loss=0.30.ckpt'
    else:
        ckpt = 'logs/span_prediction_microsoft_mdeberta-v3-base/version_0/checkpoints/checkpoint-epoch=06-val_loss=0.75.ckpt'
    for mode in ['regression','classification']:
        for grading_model in ['decision_tree', 'summation']:
            if 'task' == 'token_classification':
                context = True
            else:
                context = None
            config = Config(task=task,
                            model=MODEL,
                            train_file=TRAIN_FILE,
                            dev_file=DEV_FILE,
                            checkpoint_path=ckpt,
                            context=context,
                            device=DEVICE,
                            mode=mode,
                            grading_model=grading_model,
                            summation_th=0.7,
                            batch_size=4,
                            lr=0.01,
                            matching='fuzzy',
                            is_fixed_learner=False,
                        )
            TrainingGrading(config).run_training()



