import logging

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

import myutils as utils
from dataset import JustificationCueDataset, SpanJustificationCueDataset, GradingDataset, CustomBatchSampler
from grading_model import GradingModelTrivial, GradingModel
from model import TokenClassificationModel, SpanPredictionModel
from paraphrase_scorer import BertScorer
from preprocessor import PreprocessorTokenClassification, PreprocessorSpanPrediction
import warnings
warnings.filterwarnings("ignore")


class TrainingJustificationCueDetection:
    def __init__(self, config):
        self.config = config
        if self.config.DEVICE == 'cuda':
            num_gpus = self.config.GPUS
        else:
            num_gpus= 1
        total_batch_size = self.config.BATCH_SIZE * num_gpus
        self.EXPERIMENT_NAME = config.TASK + "_" + self.config.MODEL_NAME.replace('/', '_') +  "_bs-" + str(total_batch_size) + "_aggr-" + self.config.AGGREGATION_METHOD
        if self.config.TASK == 'token_classification':
            self. EXPERIMENT_NAME += "_context-" + str(self.config.CONTEXT)
        logger = CSVLogger("logs", name=self.EXPERIMENT_NAME)
        self.trainer = Trainer(max_epochs=self.config.NUM_EPOCHS,
                  #gradient_clip_val=0.5,
                  #accumulate_grad_batches=2,
                  callbacks=[
                      self.config.checkpoint_callback,
                      self.config.early_stop_callback,
                      self.config.lr_callback
                             ],
                  logger=logger,
                               gpus=self.config.GPUS,
                               )

    def run_training(self):
        # Load data
        rubrics = utils.load_rubrics(self.config.PATH_RUBRIC)
        if self.config.TASK == 'token_classification':
            # Load data
            train_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.TRAIN_FILE)
            dev_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.DEV_FILE)
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
            train_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.TRAIN_FILE)
            dev_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.DEV_FILE)
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
        logging.info("Start training", self.EXPERIMENT_NAME)
        self.trainer.fit(model, train_loader, val_loader)
        logging.info("End training", self.EXPERIMENT_NAME)

class TrainingGrading():

    def __init__(self, config):
        self.config = config
        self.EXPERIMENT_NAME = config.TASK + "_" + self.config.MODEL_NAME.replace('/', '_') + "_" + self.config.GRADING_MODEL
        logger = CSVLogger("logs", name=self.EXPERIMENT_NAME)
        self.trainer = Trainer(max_epochs=self.config.NUM_EPOCHS,
                               # gradient_clip_val=0.5,
                               # accumulate_grad_batches=2,
                               callbacks=[
                                   self.config.checkpoint_callback,
                                   self.config.early_stop_callback,
                                   self.config.lr_callback
                               ],
                               logger=logger,
                               gpus=self.config.GPUS,
                               num_sanity_val_steps=0,
                               )

    def run_training(self):
        rubrics = utils.load_rubrics(self.config.PATH_RUBRIC)
        if self.config.GRADING_MODEL == 'trivial':
            model = GradingModelTrivial(self.config.PATH_CHECKPOINT, rubrics)
        elif self.config.GRADING_MODEL == 'decision_tree':
            model = GradingModel(self.config.PATH_CHECKPOINT, rubrics)

        if self.config.TASK == 'token_classification':
            preprocessor = PreprocessorTokenClassification(self.config.MODEL_NAME, with_context=self.config.CONTEXT)
        elif self.config.TASK == 'span_prediction':
            preprocessor = PreprocessorSpanPrediction(self.config.MODEL_NAME, rubrics=rubrics)

        train_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.TRAIN_FILE)
        dev_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.DEV_FILE)
        training_dataset = preprocessor.preprocess(train_data)
        dev_dataset = preprocessor.preprocess(dev_data)
        # Generate dataset
        training_dataset = GradingDataset(training_dataset)
        dev_dataset = GradingDataset(dev_dataset)
        # Generate data loaders
        train_loader = DataLoader(training_dataset, batch_sampler=CustomBatchSampler(training_dataset, self.config.BATCH_SIZE))
        val_loader = DataLoader(dev_dataset, batch_sampler=CustomBatchSampler(dev_dataset, self.config.BATCH_SIZE))

        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        # Training
        logging.info("Start training", self.EXPERIMENT_NAME)
        self.trainer.fit(model, train_loader, val_loader)
        logging.info("End training", self.EXPERIMENT_NAME)