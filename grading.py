# This script finally grades the student answers
from abc import abstractmethod

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForTokenClassification
from incremental_trees.models.classification.streaming_rfc import StreamingRFC

# Define the model checkpoint
from config import Config
from dataset import GradingDataset, CustomBatchSampler
from grading_model import *
import myutils as utils
from preprocessor import *


class Grading:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.rubrics = utils.load_rubrics(self.config.PATH_RUBRIC)
        self.model = GradingModel(
            model_name=self.config.MODEL_NAME,
            checkpoint=self.config.PATH_CHECKPOINT,
            mode=self.config.MODE,
            rubrics=self.rubrics,
        )
        EXPERIMENT_NAME = "grading_" + config.MODEL_NAME + self.config.MODE
        logger = CSVLogger("logs", name=EXPERIMENT_NAME)
        self.trainer = Trainer(
            max_epochs=self.config.NUM_EPOCHS,
            callbacks=[
                self.config.checkpoint_callback,
                self.config.early_stop_callback
            ],
            logger=logger,
            num_sanity_val_steps=0,
            )

    def training(self):
        train_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.TRAIN_FILE)
        dev_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.DEV_FILE)
        if self.config.TASK == 'token_classification':
            preprocessor = GradingPreprocessorTokenClassification(self.tokenizer, with_context=self.config.CONTEXT)
            training_dataset = preprocessor.preprocess(train_data)
            dev_dataset = preprocessor.preprocess(dev_data)
        elif self.config.TASK == 'span_prediction':
            preprocessor = GradingPreprocessorSpanPrediction(self.tokenizer, self.rubrics)
            training_dataset = preprocessor.preprocess(train_data)
            dev_dataset = preprocessor.preprocess(dev_data)
        training_dataset = GradingDataset(training_dataset)
        dev_dataset = GradingDataset(dev_dataset)
        train_loader = DataLoader(training_dataset,
                                  batch_sampler=CustomBatchSampler(training_dataset, self.config.BATCH_SIZE))
        val_loader = DataLoader(dev_dataset, batch_sampler=CustomBatchSampler(dev_dataset, self.config.BATCH_SIZE))
        self.trainer.fit(self.model, train_loader, val_loader)

    def test(self):
        self.model = self.model.load_from_checkpoint(self.config.CHECKPOINT_PATH)
        test_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.TEST_FILE)
        if self.config.TASK == 'token_classification':
            preprocessor = GradingPreprocessorTokenClassification(with_context=self.config.CONTEXT)
            test_dataset = preprocessor.preprocess(test_data)

        elif self.config.TASK == 'span_prediction':
            preprocessor = GradingPreprocessorSpanPrediction(self.rubrics)
            test_dataset = preprocessor.preprocess(test_data)
        test_dataset = GradingDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_sampler=CustomBatchSampler(test_dataset, self.config.BATCH_SIZE))
        self.trainer.test(self.model, test_loader)

if __name__ == '__main__':
    config = Config(task='token_classification',
                    gpus=2,
                    #checkpoint_path='logs/span_prediction_SpanBERT_spanbert-base-cased_bs-8/version_0/checkpoints/checkpoint-epoch=00-val_loss=6.24.ckpt',
                    checkpoint_path='logs/token_classification_SpanBERT_spanbert-base-cased_bs-8_context-True/version_0/checkpoints/checkpoint-epoch=05-val_loss=0.60.ckpt',
                    dev_file='dev_dataset_aligned_labels_SpanBERT_spanbert-base-cased.json',
                    train_file='training_dataset_aligned_labels_SpanBERT_spanbert-base-cased.json',
                    model='SpanBERT/spanbert-base-cased'
                    )
    Grading(config).training()