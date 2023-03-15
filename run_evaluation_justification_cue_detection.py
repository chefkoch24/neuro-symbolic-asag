import logging

import torch

from config import Config
from evaluation import EvaluationJustificationCueDetection
from training import TrainingJustificationCueDetection

TASK = 'token_classification'
CHECKPOINT_PATH = 'logs/token_classification_distilbert-base-multilingual-cased_bs-8_aggr-lfs_sum_context-True/version_0/checkpoints/checkpoint-epoch=00-val_loss=0.67.ckpt'
MODEL = 'distilbert-base-multilingual-cased'
CONTEXT = True
TEST_FILE = 'dev_dataset_aligned_labels_distilbert-base-multilingual-cased_lfs_sum.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AGGREGATION_METHOD = 'lfs_sum'

config = Config(task=TASK,
                device=DEVICE,
                model=MODEL,
                aggregation_method=AGGREGATION_METHOD,
                context=CONTEXT,
                test_file=TEST_FILE,
                checkpoint_path=CHECKPOINT_PATH,
                )
EvaluationJustificationCueDetection(config).run_evaluation()
