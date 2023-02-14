# Imports
from model import TokenClassificationModel, SoftLabelTokenClassificationModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import logging
import torch
import config
import utils
logging.basicConfig(level=logging.ERROR)
from dataset import JustificationCueDataset
#torch.distributed.is_available=False
import  tqdm
from transformers import AutoModelForTokenClassification

#Set seed
#torch.manual_seed(config.SEED)
#torch.cuda.manual_seed_all(config.SEED)

# Load data
training_data = utils.load_json(config.PATH_DATA + '/' + 'training_dataset.json')
dev_data = utils.load_json(config.PATH_DATA + '/' + 'dev_dataset.json')
training_dataset = JustificationCueDataset(training_data[0:8])
dev_dataset = JustificationCueDataset(dev_data[0:8])

rubrics = utils.load_rubrics(config.PATH_RUBRIC)


# Training
train_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
#model = AutoModelForTokenClassification.from_pretrained(config.MODEL_NAME)
#model = SoftLabelTokenClassificationModel(config.MODEL_NAME, rubrics)
model = TokenClassificationModel(config.MODEL_NAME, rubrics)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

trainer = Trainer(max_epochs=config.NUM_EPOCHS)
trainer.fit(model, train_loader, val_loader)

# plain pytorch training loop
#for e in range(config.NUM_EPOCHS):
#    train_loss = []
#    for batch in tqdm.tqdm(train_loader):
#        optimizer.zero_grad()
#        _, loss = model.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
#        loss.backward()
#        optimizer.step()
#        train_loss.append(loss.item())
#    tqdm.tqdm.write(f'Epoch: {e}, Loss: {sum(train_loss) / len(train_loss)}')

    # Validation
#    val_loss = []
#    model.eval()
#    optimizer.zero_grad()
#    for batch in tqdm.tqdm(val_loader):
#        outputs, loss = model.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
#        val_loss.append(loss.item())
#    compute_metrics(outputs, batch)
#    tqdm.tqdm.write(f'Epoch: {e}, Loss: {sum(val_loss) / len(val_loss)}')