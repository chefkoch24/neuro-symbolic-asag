from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from config import Config
import myutils as utils
from dataset import SpanJustificationCueDataset, JustificationCueDataset
from justification_cue_model import SpanPredictionModel, TokenClassificationModel

#SHARED
config = Config()
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

# SPAN PREDICTION
validation_file= 'dev_dataset_span_prediction_distilbert-base-multilingual-cased.json'
span_checkpoint_path = 'logs/span_prediction_distilbert-base-multilingual-cased/version_8/checkpoints/checkpoint-epoch=01-val_loss=3.52.ckpt'

validation_data = utils.load_json(config.PATH_DATA + '/' + validation_file)
validation_dataset = SpanJustificationCueDataset(validation_data)
validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
model = SpanPredictionModel.load_from_checkpoint(span_checkpoint_path)

trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  callbacks=[config.checkpoint_callback, config.early_stop_callback],
                  )
trainer.test(model, validation_loader)

# TOKEN CLASSIFICATION
validation_file = 'training_dataset_distilbert-base-multilingual-cased_context-False.json'
token_checkpoint_path = 'logs/justification_cue_distilbert-base-multilingual-cased_context-False/version_7/checkpoints/checkpoint-epoch=04-val_loss=0.64.ckpt'

validation_data = utils.load_json(config.PATH_DATA + '/' + validation_file)
validation_dataset = JustificationCueDataset(validation_data)
validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
model = TokenClassificationModel.load_from_checkpoint(token_checkpoint_path)
trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  callbacks=[config.checkpoint_callback, config.early_stop_callback],
                  )
trainer.test(model, validation_loader)

