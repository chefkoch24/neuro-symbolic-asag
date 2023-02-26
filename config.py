# Here is all the stuff configured that is needed across scripts
import argparse
import spacy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# For local development
#args.train_file = 'training_dataset_distilbert-base-multilingual-cased_context-False.json'
#args.dev_file = 'dev_dataset_distilbert-base-multilingual-cased_context-False.json'
#args.model='distilbert-base-multilingual-cased'

MODEL_NAME = "distilbert-base-multilingual-cased"
ANNOTATED_TRAIN_FILE = 'train_labeled_data_sum.json'
ANNOTATED_DEV_FILE = 'dev_labeled_data_sum.json'
TEST_FILE = None
NUM_EPOCHS = 8
BATCH_SIZE = 8
CONTEXT = True
MAX_LEN = 512
SEED = 42
TOKENIZER_NAME = MODEL_NAME
PATH_DATA = "data"
PATH_RUBRIC = "data/rubrics.json"
PATH_RAW_DATA = "input/safdataset/training"
PATH_RAW_RUBRIC = "input/rubrics"
PATH_CHECKPOINT = "checkpoints"

nlp = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")

checkpoint_callback = ModelCheckpoint(
    filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    verbose=True,
    monitor='val_loss',
    mode='min',

)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)