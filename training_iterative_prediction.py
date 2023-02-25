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

# Load data
training_data = utils.load_json(config.PATH_DATA + '/' + config.TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.DEV_FILE)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

# Create dataset and dataloader
training_dataset = IterativeJustificationCueDataset(training_data)
dev_dataset = IterativeJustificationCueDataset(dev_data)


train_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = TokenClassificationModel(config.MODEL_NAME, rubrics=rubrics)

EXPERIMENT_NAME = "justification_cue_iterative" + "_" + config.MODEL_NAME
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

