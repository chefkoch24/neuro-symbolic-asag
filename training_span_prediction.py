import numpy as np
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import myutils as utils
import config
from dataset import SpanJustificationCueDataset
from model import SpanPredictionModel


#Set seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
# Load data
training_data = utils.load_json(config.PATH_DATA + '/' + config.TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.DEV_FILE)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

training_dataset = SpanJustificationCueDataset(training_data)
dev_dataset = SpanJustificationCueDataset(dev_data)
# Training
train_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
model = SpanPredictionModel(config.MODEL_NAME)

EXPERIMENT_NAME = "span_prediction" + "_" + config.MODEL_NAME + "_batch_" + str(config.BATCH_SIZE)
logger = CSVLogger("logs", name=EXPERIMENT_NAME)
trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  #gradient_clip_val=0.5,
                  #accumulate_grad_batches=2,
                  #auto_scale_batch_size='power',
                  callbacks=[config.checkpoint_callback, config.early_stop_callback],
                  logger=logger,
                  )
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)