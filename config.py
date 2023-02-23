# Here is all the stuff configured that is needed across scripts
MODEL_NAME = "distilroberta-base"
#MODEL_NAME = "distilbert-base-multilingual-cased"
#MODEL_NAME = 'xlm-roberta-base'
#MODEL_NAME = 'SpanBERT/spanbert-base-cased'
TOKENIZER_NAME = MODEL_NAME
SEED = 42
NUM_EPOCHS = 8
PATH_DATA = "data"
PATH_RUBRIC = "data/rubrics.json"
PATH_RAW_DATA = "input/safdataset/training"
PATH_RAW_RUBRIC = "input/rubrics"
PATH_CHECKPOINT = "checkpoints"
MAX_LEN = 512
BATCH_SIZE = 8

# Imports
import spacy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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