import torch

from config import Config
from prediction import PredictSpan, PredictToken

TASK = 'token_classification'
CHECKPOINT_PATH = 'logs/token_classification_distilbert-base-multilingual-cased_bs-8_aggr-lfs_sum_context-True/version_0/checkpoints/checkpoint-epoch=00-val_loss=0.67.ckpt'
MODEL = 'distilbert-base-multilingual-cased'
CONTEXT = True
TEST_FILE = 'dev_dataset_aligned_labels_distilbert-base-multilingual-cased_lfs_sum.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if TASK == 'span_prediction':
    config = Config(task='span_prediction',
                    model=MODEL,
                    test_file=TEST_FILE,
                    checkpoint_path=CHECKPOINT_PATH,
                    device=DEVICE,
                    )
    prediction = PredictSpan(config)
elif TASK == 'token_classification':
    config = Config(task='token_classification',
                    model=MODEL,
                    test_file=TEST_FILE,
                    checkpoint_path=CHECKPOINT_PATH,
                    context=CONTEXT,
                    device=DEVICE,
                    )
    prediction = PredictToken(config)
prediction.predict()

