# runs all experiments for the justification cue detection task
import subprocess
import logging
from config import Config
from training import Training

for task in ['span_prediction', 'token_classification']:
    config = Config()
    config.TASK = task
    for model in ["SpanBERT/spanbert-base-cased" , "distilbert-base-multilingual-cased", ]:
        config.MODEL_NAME = model
        config.ALIGNED_TRAIN_FILE = 'training_dataset_aligned_labels_' + config.MODEL_NAME.replace('/', '_') + '.json'
        config.ALIGNED_DEV_FILE = "dev_dataset_aligned_labels_" + config.MODEL_NAME.replace('/', '_') + ".json"
        for context in [True, False]:
            config.CONTEXT = context
            for bs in [8, 16, 32]:
                config.BATCH_SIZE = bs
                training = Training(config)
                training.run_training()