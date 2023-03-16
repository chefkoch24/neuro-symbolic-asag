# Here is all the stuff configured that is needed across scripts
import argparse
import spacy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

class Config:
    def __init__(self,
                 model="distilbert-base-multilingual-cased",
                 train_file='train_file.json',
                 dev_file='dev_file.json',
                 test_file='test_file.json',
                 task='token_classification',
                 num_epochs=8,
                 batch_size=8,
                 context=False,
                 max_len=512,
                 seed=42,
                 gpus=1,
                 lr=None,
                 checkpoint_path=None,
                 aggregation_method='sum',
                 excluded_lfs=[],
                 mode='classification',
                 grading_model='decision_tree',
                 device='cpu'
                 ):
        self.AGGREGATION_METHOD = aggregation_method
        self.EXCLUDED_LFS= excluded_lfs
        self.MODEL_NAME = model
        self.TRAIN_FILE = train_file
        self.DEV_FILE = dev_file
        self.TEST_FILE = test_file
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.CONTEXT = context
        self.MAX_LEN = max_len
        self.SEED = seed
        self.TOKENIZER_NAME = self.MODEL_NAME
        self.PATH_CHECKPOINT = checkpoint_path
        self.TASK = task
        self.LR = lr
        self.GPUS = gpus
        self.MODE = mode
        self.GRADING_MODEL = grading_model
        self.DEVICE = device
        # FIXED SETTINGS AND PATHS
        self.PATH_DATA = "data"
        self.PATH_RUBRIC = "data/rubrics.json"
        self.PATH_RAW_DATA = "input/safdataset/training"
        self.PATH_RAW_RUBRIC = "input/rubrics"
        self.PATH_RESULTS = "results"
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp_de = spacy.load("de_core_news_lg")
        self.lr_callback = LearningRateMonitor(logging_interval='epoch', log_momentum=True)
        self.checkpoint_callback = ModelCheckpoint(
            filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
        )

        self.early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=3,
            mode='min'
        )