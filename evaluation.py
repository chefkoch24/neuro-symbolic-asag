import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import myutils as utils
from dataset import JustificationCueDataset, SpanJustificationCueDataset
from model import TokenClassificationModel, SpanPredictionModel
from paraphrase_scorer import BertScorer
from preprocessor import PreprocessorTokenClassification, PreprocessorSpanPrediction


class EvaluationJustificationCueDetection:
    def __init__(self, config):
        self.config = config
        self.trainer = Trainer()

    def run_evaluation(self):
        # Load data
        rubrics = utils.load_rubrics(self.config.PATH_RUBRIC)
        if self.config.TASK == 'token_classification':
            # Load data
            test_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.TEST_FILE)
            # Preprocess data
            preprocessor = PreprocessorTokenClassification(self.config.MODEL_NAME, with_context=self.config.CONTEXT)
            test_dataset = preprocessor.preprocess(test_data)
            # Generate dataset
            test_dataset = JustificationCueDataset(test_dataset)
            # Generate data loaders
            test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
            model = TokenClassificationModel.load_from_checkpoint(self.config.PATH_CHECKPOINT)
        elif self.config.TASK == 'span_prediction':
            # Load data
            test_data = utils.load_json(self.config.PATH_DATA + '/' + self.config.TEST_FILE)
            # Preprocess data
            scorer = BertScorer()
            preprocessor = PreprocessorSpanPrediction(self.config.MODEL_NAME, scorer=scorer, rubrics=rubrics)
            test_dataset = preprocessor.preprocess(test_data)
            # Generate dataset
            test_dataset = SpanJustificationCueDataset(test_dataset)
            # Generate data loaders
            test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
            model = SpanPredictionModel.load_from_checkpoint(self.config.PATH_CHECKPOINT)
        # Set seed
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        # Training
        self.trainer.test(model, test_loader)

class EvaluationGrading:
    def __init__(self, config):
        self.config = config
        self.trainer = Trainer()

    def run_evaluation(self):
        pass