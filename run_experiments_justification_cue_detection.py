# runs all experiments for the justification cue detection task
import logging

import torch

from config import Config
from training import TrainingJustificationCueDetection

for task in ['token_classification', 'span_prediction']:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config(task=task, device=device, gpus=1)
    for model in ["distilbert-base-multilingual-cased", "SpanBERT/spanbert-base-cased", ]:
        for aggregation in ['lfs_sum', 'hmm']:
            config.AGGREGATION_METHOD = aggregation
            config.MODEL_NAME = model
            config.TRAIN_FILE = 'training_dataset_aligned_labels_' + config.MODEL_NAME.replace('/', '_') + '_' + aggregation + '.json'
            config.DEV_FILE = "dev_dataset_aligned_labels_" + config.MODEL_NAME.replace('/', '_') + '_' + aggregation +  ".json"
            for context in [True, False]:
                config.CONTEXT = context
                for bs in [8, 16]:
                    config.BATCH_SIZE = bs
                    TrainingJustificationCueDetection(config).run_training()