# Here is all the stuff configured that is needed across scripts
import argparse
import spacy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

parser=argparse.ArgumentParser()
parser.add_argument("--model", help="Name of the pretrained model", default="distilbert-base-multilingual-cased")
parser.add_argument("--train_file", help="train file")
parser.add_argument("--dev_file", help="dev file")
parser.add_argument("--test_file", help="test file", default=None)
parser.add_argument("--max_len", help="max length of the input", default=512)
parser.add_argument("--batch_size", help="batch size", default=8)
parser.add_argument("--num_epochs", help="number of epochs", default=8)
parser.add_argument("--seed", help="seed int", default=42)
parser.add_argument("--context", help="with context or not", default='False')

args=parser.parse_args()
if args.context == 'True':
    args.context = True
else:
    args.context = False

# For local development
args.train_file = 'training_dataset.json'
args.dev_file = 'dev_dataset.json'
args.model='distilbert-base-multilingual-cased'

MODEL_NAME = args.model
TOKENIZER_NAME = MODEL_NAME
SEED = args.seed
NUM_EPOCHS = args.num_epochs
PATH_DATA = "data"
PATH_RUBRIC = "data/rubrics.json"
PATH_RAW_DATA = "input/safdataset/training"
PATH_RAW_RUBRIC = "input/rubrics"
PATH_CHECKPOINT = "checkpoints"
MAX_LEN = args.max_len
BATCH_SIZE = args.batch_size
TRAIN_FILE = args.train_file
DEV_FILE = args.dev_file
TEST_FILE = args.test_file
CONTEXT = args.context

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