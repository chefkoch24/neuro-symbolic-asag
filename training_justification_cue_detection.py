# Imports
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import TokenClassificationModel, TokenClassificationModelBinary
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
#model = TokenClassificationModelBinary(config.MODEL_NAME, rubrics)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

checkpoint_callback = ModelCheckpoint(
    dirpath=config.PATH_CHECKPOINT,
    filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
)

early_stop_callback = EarlyStopping(
    monitor='loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)
logger = CSVLogger("logs", name="my_logs")
trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  gradient_clip_val=0.5,
                  accumulate_grad_batches=2,
                  #auto_scale_batch_size='power',
                  #callbacks=[checkpoint_callback, early_stop_callback],
                  logger=logger)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)

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