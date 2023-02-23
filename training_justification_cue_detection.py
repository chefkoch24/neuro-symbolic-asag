# Imports
import argparse
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model import TokenClassificationModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import logging
import torch
import config
import myutils as utils
logging.basicConfig(level=logging.ERROR)
from dataset import JustificationCueDataset
import warnings
warnings.filterwarnings("ignore")

#Set seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

parser=argparse.ArgumentParser()

parser.add_argument("--model", help="Name of the pretrained model")
parser.add_argument("--train_file", help="train file")
parser.add_argument("--dev_file", help="dev file")
parser.add_argument("--test_file", help="test file")
parser.add_argument("--context", help="with context or not")
args=parser.parse_args()

if args.context == 'True':
    args.context = True
else:
    args.context = False

# Load data
training_data = utils.load_json(config.PATH_DATA + '/' + args.train_file)
dev_data = utils.load_json(config.PATH_DATA + '/' + args.dev_file)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

training_dataset = JustificationCueDataset(training_data)
dev_dataset = JustificationCueDataset(dev_data)
# Training
train_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
model = TokenClassificationModel(args.model, rubrics)


EXPERIMENT_NAME = "justification_cue" + "_" + args.model + "_context-" + str(args.context)
logger = CSVLogger("logs", name=EXPERIMENT_NAME)
trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  #gradient_clip_val=0.5,
                  #accumulate_grad_batches=2,
                  #auto_scale_batch_size='power',
                  callbacks=[
                      config.checkpoint_callback,
                      config.early_stop_callback
                             ],
                  logger=logger)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)