import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

import myutils as utils
from dataset import JustificationCueDataset, SpanJustificationCueDataset
from model import TokenClassificationModel, SpanPredictionModel
from paraphrase_scorer import BertScorer
from preprocessor import PreprocessorTokenClassification, PreprocessorSpanPrediction
import warnings
warnings.filterwarnings("ignore")


class Training:
    def __init__(self, config):
        self.config = config
        EXPERIMENT_NAME = config.TASK + "_" + self.config.MODEL_NAME.replace('/', '_') +  "_bs-" + str(self.config.BATCH_SIZE)
        if config.TASK == 'token_classification':
            EXPERIMENT_NAME += "_context-" + str(self.config.CONTEXT)
        logger = CSVLogger("logs", name=EXPERIMENT_NAME)
        self.trainer = Trainer(max_epochs=self.config.NUM_EPOCHS,
                  #gradient_clip_val=0.5,
                  #accumulate_grad_batches=2,
                  callbacks=[
                      self.config.checkpoint_callback,
                      self.config.early_stop_callback,
                      self.config.lr_callback
                             ],
                  logger=logger)

    def run_training(self):
        # Load data
        rubrics = utils.load_rubrics(self.config.PATH_RUBRIC)
        if self.config.TASK == 'token_classification':
            # Load data
            train_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.ALIGNED_TRAIN_FILE)
            dev_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.ALIGNED_DEV_FILE)
            # Preprocess data
            preprocessor = PreprocessorTokenClassification(self.config.MODEL_NAME, with_context=self.config.CONTEXT)
            training_dataset = preprocessor.preprocess(train_data)
            dev_dataset = preprocessor.preprocess(dev_data)
            # Generate dataset
            training_dataset = JustificationCueDataset(training_dataset)
            dev_dataset = JustificationCueDataset(dev_dataset)
            # Generate data loaders
            train_loader = DataLoader(training_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(dev_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
            model = TokenClassificationModel(self.config.MODEL_NAME)
        elif self.config.TASK == 'span_prediction':
            # Load data
            train_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.ALIGNED_TRAIN_FILE)
            dev_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.ALIGNED_DEV_FILE)
            # Preprocess data
            scorer = BertScorer()
            preprocessor = PreprocessorSpanPrediction(self.config.MODEL_NAME, scorer=scorer, rubrics=rubrics)
            training_dataset = preprocessor.preprocess(train_data)
            dev_dataset = preprocessor.preprocess(dev_data)
            # Generate dataset
            training_dataset = SpanJustificationCueDataset(training_dataset)
            dev_dataset = SpanJustificationCueDataset(dev_dataset)
            # Generate data loaders
            train_loader = DataLoader(training_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(dev_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

            model = SpanPredictionModel(self.config.MODEL_NAME)
        # Set seed
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        # Training
        self.trainer.fit(model, train_loader, val_loader)
        self.trainer.test(model, val_loader)

class TrainingGrading:
    def __init__(self, config):
        self.config = config
    def run_training(self):
        pass