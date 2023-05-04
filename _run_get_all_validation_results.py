import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

import myutils as utils
from config import Config
from dataset import GradingDataset, CustomBatchSampler
from grading_model import GradingModel
from preprocessor import GradingPreprocessor

config = Config(
    task='token_classification',
    model='microsoft/mdeberta-v3-base',
    dev_file='dev_dataset.json',
    context=True,
)

checkpoints = [
    'grading_token_classification_2023-04-04_22-42',
    'grading_token_classification_2023-04-05_02-42',
    'grading_token_classification_2023-04-04_23-22',
    'grading_token_classification_2023-04-05_03-22',
    'grading_token_classification_2023-04-05_00-02',
    'grading_token_classification_2023-04-05_04-02',
    'grading_token_classification_2023-04-05_00-42',
    'grading_token_classification_2023-04-05_04-42',
    'grading_token_classification_2023-04-05_01-22',
    'grading_token_classification_2023-04-05_05-22',
    'grading_token_classification_2023-04-05_02-02',
    'grading_token_classification_2023-04-05_06-02',
    'grading_token_classification_2023-04-05_06-42',
    'grading_token_classification_2023-04-05_10-43',
    'grading_token_classification_2023-04-05_07-22',
    'grading_token_classification_2023-04-05_11-24',
    'grading_token_classification_2023-04-05_08-02',
    'grading_token_classification_2023-04-05_12-04',
    'grading_token_classification_2023-04-05_08-42',
    'grading_token_classification_2023-04-05_12-44',
    'grading_token_classification_2023-04-05_09-22',
    'grading_token_classification_2023-04-05_13-24',
    'grading_token_classification_2023-04-05_10-02',
    'grading_token_classification_2023-04-05_14-04',
]
seed_everything(config.SEED, workers=True)
# get all validation result from hyperparameter search
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
preprocessor = GradingPreprocessor(config.MODEL_NAME, with_context=config.CONTEXT, rubrics=rubrics)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.DEV_FILE)
dev_dataset = preprocessor.preprocess(dev_data)
dev_dataset = GradingDataset(dev_dataset)
val_loader = DataLoader(dev_dataset, batch_sampler=CustomBatchSampler(dev_dataset, config.BATCH_SIZE, seed=config.SEED))
trainer = Trainer(
    deterministic=True,
    accelerator='cuda' if torch.cuda.is_available() else 'cpu',
    gpus=1 if torch.cuda.is_available() else 0,
    #fast_dev_run=True,
)
result = {}
for path in checkpoints:
    CHECKPOINT_PATH_SYMBOLIC_MODELS = 'logs/' + path + '/symbolic_models/'
    checkpoint = os.listdir('logs/' + path + '/version_0/checkpoints/')[0]
    num_epoch = checkpoint.split('=')[1].split('-')[0][1]
    sym_model = 'epoch_' + str(num_epoch)

    symbolic_models = utils.load_symbolic_models(CHECKPOINT_PATH_SYMBOLIC_MODELS + sym_model, rubrics)
    model = GradingModel.load_from_checkpoint('logs/' + path + '/version_0/checkpoints/' + checkpoint, symbolic_models=symbolic_models)
    val_res = trainer.validate(model, val_loader)
    result[path] = val_res

utils.save_json(result, config.PATH_RESULTS, 'hyperparamter_final_res.json')