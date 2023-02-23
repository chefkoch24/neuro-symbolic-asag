# Imports
import argparse
from pytorch_lightning.loggers import CSVLogger
from model import TokenClassificationModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import logging
import torch
import config
import myutils as utils
import warnings
from dataset import IterativeJustificationCueDataset

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

#Set seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

parser=argparse.ArgumentParser()

parser.add_argument("--model", help="Name of the pretrained model")
parser.add_argument("--train_file", help="train file")
parser.add_argument("--dev_file", help="dev file")
parser.add_argument("--test_file", help="test file")
args=parser.parse_args()

# Load data
training_data = utils.load_json(config.PATH_DATA + '/' + args.train_file)
dev_data = utils.load_json(config.PATH_DATA + '/' + args.dev_file)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

# Create dataset and dataloader
training_dataset = IterativeJustificationCueDataset(training_data)
dev_dataset = IterativeJustificationCueDataset(dev_data)


train_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = TokenClassificationModel(args.model, rubrics=rubrics)

EXPERIMENT_NAME = "justification_cue_iterative" + "_" + args.model
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

