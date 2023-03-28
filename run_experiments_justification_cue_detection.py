# runs all experiments for the justification cue detection task

import torch

from config import Config
from training import TrainingJustificationCueDetection

for task in ['span_prediction', 'token_classification', ]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for model in ["microsoft/mdeberta-v3-base", "distilbert-base-multilingual-cased"]:
            train_file = 'training_aligned_labels_' + model.replace('/', '_') + '.json'
            dev_file = "dev_aligned_labels_" + model.replace('/', '_') + ".json"
            if task == 'token_classification':
               for context in [True, False]:
                   config = Config(
                       task=task,
                       model=model,
                       train_file=train_file,
                       dev_file=dev_file,
                       device=device,
                       gpus=1,
                       batch_size=8,
                       context=context
                   )
                   TrainingJustificationCueDetection(config).run_training()
            else:
                config = Config(
                    task=task,
                    model=model,
                    train_file=train_file,
                    dev_file=dev_file,
                    device=device,
                    gpus=1,
                    batch_size=4
                )
                TrainingJustificationCueDetection(config).run_training()