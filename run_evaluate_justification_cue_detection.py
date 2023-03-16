from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import config
import myutils as utils
from dataset import SpanJustificationCueDataset, JustificationCueDataset
from model import SpanPredictionModel, TokenClassificationModel

#SHARED
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

# SPAN PREDICTION
test_file= 'dev_dataset_span_prediction_distilbert-base-multilingual-cased.json'
span_checkpoint_path = 'logs/span_prediction_distilbert-base-multilingual-cased/version_8/checkpoints/checkpoint-epoch=01-val_loss=3.52.ckpt'

test_data = utils.load_json(config.PATH_DATA + '/' + test_file)
test_dataset = SpanJustificationCueDataset(test_data[0:8])
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
model = SpanPredictionModel.load_from_checkpoint(span_checkpoint_path)

trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  callbacks=[config.checkpoint_callback, config.early_stop_callback],
                  )
trainer.test(model, test_loader)

# TOKEN CLASSIFICATION
test_file= 'training_dataset_distilbert-base-multilingual-cased_context-False.json'
token_checkpoint_path = 'logs/justification_cue_distilbert-base-multilingual-cased_context-False/version_7/checkpoints/checkpoint-epoch=04-val_loss=0.64.ckpt'

test_data = utils.load_json(config.PATH_DATA + '/' + test_file)
test_dataset = JustificationCueDataset(test_data[0:8])
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
model = TokenClassificationModel.load_from_checkpoint(token_checkpoint_path)
trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  callbacks=[config.checkpoint_callback, config.early_stop_callback],
                  )
trainer.test(model, test_loader)

