# Imports
from model import TokenClassificationModel
from utils import load_rubrics
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import logging
import torch
logging.basicConfig(level=logging.ERROR)

#Set seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load data
training_dataset = torch.load(PATH_DATA+'training_dataset.pt')
dev_dataset = torch.load(PATH_DATA+'dev_dataset.pt')

rubrics = load_rubrics(PATH_RUBRIC)

# Training
trainer = Trainer(max_epochs=NUM_EPOCHS, progress_bar_refresh_rate=1)
train_loader = DataLoader(training_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)
model = TokenClassificationModel(model_name=MODEL_NAME, rubrics=rubrics)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.save_checkpoint(PATH_DATA + MODEL_NAME+"-justification-cue-model.pt")