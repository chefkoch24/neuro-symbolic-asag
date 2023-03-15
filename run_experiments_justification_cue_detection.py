# runs all experiments for the justification cue detection task
import logging

import torch

from config import Config
from training import TrainingJustificationCueDetection

for task in ['token_classification', 'span_prediction']:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config(task=task, device=device, gpus=2)
    for model in ["SpanBERT/spanbert-base-cased", "distilbert-base-multilingual-cased"]:
        for aggregation in ['lfs_sum', 'hmm']:
            config.AGGREGATION_METHOD = aggregation
            config.MODEL_NAME = model
            config.TRAIN_FILE = 'training_dataset_aligned_labels_' + config.MODEL_NAME.replace('/', '_') + '_' + aggregation + '.json'
            config.DEV_FILE = "dev_dataset_aligned_labels_" + config.MODEL_NAME.replace('/', '_') + '_' + aggregation +  ".json"
            for context in [True, False]:
                config.CONTEXT = context
                config.BATCH_SIZE = 4
                TrainingJustificationCueDetection(config).run_training()