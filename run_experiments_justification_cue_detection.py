# runs all experiments for the justification cue detection task
import logging

import torch

from config import Config
from training import TrainingJustificationCueDetection

for task in ['span_prediction']:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for model in ["distilbert-base-multilingual-cased", "SpanBERT/spanbert-base-cased", ]:
        for aggregation in ['lfs_sum', 'hmm']:
            train_file = 'training_dataset_aligned_labels_' + model.replace('/', '_') + '_' + aggregation + '.json'
            dev_file = "dev_dataset_aligned_labels_" + model.replace('/', '_') + '_' + aggregation + ".json"
            for context in [True, False]:
                config = Config(
                    task=task,
                    model=model,
                    train_file=train_file,
                    dev_file=dev_file,
                    aggregation_method=aggregation,
                    context=context,
                    device=device,
                    gpus=1,
                    batch_size=8
                )
                TrainingJustificationCueDetection(config).run_training()