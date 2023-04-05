import logging
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from dataset import JustificationCueDataset, SpanJustificationCueDataset, GradingDataset, CustomBatchSampler
from grading_model import GradingModel
from justification_cue_model import TokenClassificationModel, SpanPredictionModel
from paraphrase_scorer import BertScorer
from preprocessor import *
import warnings
warnings.filterwarnings("ignore")


class TrainingJustificationCueDetection:
    def __init__(self, config):
        self.config = config
        seed_everything(self.config.SEED, workers=True)
        if self.config.DEVICE == 'cuda':
            num_gpus = self.config.GPUS
        else:
            num_gpus= 1
        if self.config.TASK == 'token_classification':
            self.EXPERIMENT_NAME = utils.get_experiment_name([self.config.TASK, self.config.MODEL_NAME, self.config.CONTEXT])
        else:
            self.EXPERIMENT_NAME = utils.get_experiment_name([self.config.TASK, self.config.MODEL_NAME])
        gradient_accumulation_steps = 2
        logging_hyperparams = {
            'batch_size': self.config.BATCH_SIZE,
            'num_gpus': num_gpus,
            'device': self.config.DEVICE,
            'context': self.config.CONTEXT,
            'model_name': self.config.MODEL_NAME,
            'max_length' : self.config.MAX_LEN,
            'seed': self.config.SEED,
            'task': self.config.TASK,
            'gradient_accumulation_steps': gradient_accumulation_steps
        }
        utils.save_json(logging_hyperparams, path='logs/' + self.EXPERIMENT_NAME, file_name= 'hyperparams.json')
        logger = CSVLogger("logs", name=self.EXPERIMENT_NAME)
        self.trainer = Trainer(
            accumulate_grad_batches=gradient_accumulation_steps,
            max_epochs=self.config.NUM_EPOCHS,
            callbacks=[
                      self.config.checkpoint_callback,
                      self.config.early_stop_callback,
                      self.config.lr_callback
                             ],
            logger=logger,
            gpus=self.config.GPUS,
            accelerator=self.config.DEVICE,
            deterministic=True,
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

class TrainingGrading:

    def __init__(self, config):
        self.config = config
        seed_everything(self.config.SEED, workers=True)
        current_time = datetime.now()
        current_time_string = current_time.strftime("%Y-%m-%d_%H-%M")
        self.EXPERIMENT_NAME = utils.get_experiment_name(['grading', self.config.TASK, current_time_string])
        gradient_accumulation_steps = 2
        logging_hyperparams = {
            'batch_size': self.config.BATCH_SIZE,
            'num_gpus': self.config.GPUS,
            'device': self.config.DEVICE,
            'context': self.config.CONTEXT,
            'model_name': self.config.MODEL_NAME,
            'max_length': self.config.MAX_LEN,
            'seed': self.config.SEED,
            'task': self.config.TASK,
            'mode': self.config.MODE,
            'grading_model': self.config.GRADING_MODEL,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'matching': self.config.MATCHING,
            'is_fixed_learner': self.config.IS_FIXED_LEARNER,
            'lr': self.config.LR,
            'summation_th': self.config.SUMMATION_TH,
        }
        print(logging_hyperparams)
        utils.save_json(logging_hyperparams, path='logs/' + self.EXPERIMENT_NAME, file_name='hyperparams.json')
        logger = CSVLogger("logs", name=self.EXPERIMENT_NAME)
        self.trainer = Trainer(max_epochs=self.config.NUM_EPOCHS,
                               # gradient_clip_val=0.5,
                               accumulate_grad_batches=gradient_accumulation_steps,
                               callbacks=[
                                   self.config.checkpoint_callback,
                                   self.config.early_stop_callback,
                                   self.config.lr_callback
                               ],
                               logger=logger,
                               gpus=self.config.GPUS,
                               num_sanity_val_steps=0,
                               accelerator=self.config.DEVICE,
                               deterministic=True,
                               )

    def run_training(self):
        rubrics = utils.load_rubrics(self.config.PATH_RUBRIC)
        model = GradingModel(
            self.config.PATH_CHECKPOINT,
            rubrics,
            self.config.MODEL_NAME,
            mode=self.config.MODE,
            task=self.config.TASK,
            learning_strategy=self.config.GRADING_MODEL,
            summation_th=self.config.SUMMATION_TH,
            matching=self.config.MATCHING,
            lr=self.config.LR,
            is_fixed_learner=self.config.IS_FIXED_LEARNER,
            experiment_name=self.EXPERIMENT_NAME,
            seed=self.config.SEED,
        )
        preprocessor = GradingPreprocessor(self.config.MODEL_NAME, with_context=self.config.CONTEXT, rubrics=rubrics)
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
        # Training
        logging.info("Start training", self.EXPERIMENT_NAME)
        self.trainer.fit(model, train_loader, val_loader)
        logging.info("End training", self.EXPERIMENT_NAME)