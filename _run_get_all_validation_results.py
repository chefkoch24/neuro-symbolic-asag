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
    'grading_token_classification_2023-05-04_11-25',
    'grading_token_classification_2023-05-04_12-07',
    'grading_token_classification_2023-05-04_12-49',
    'grading_token_classification_2023-05-04_13-31',
    'grading_token_classification_2023-05-04_14-12',
    'grading_token_classification_2023-05-04_14-12',
    'grading_token_classification_2023-05-04_15-36',
    'grading_token_classification_2023-05-04_16-18',
    'grading_token_classification_2023-05-04_17-00',
    'grading_token_classification_2023-05-04_17-42',
    'grading_token_classification_2023-05-04_18-24',
    'grading_token_classification_2023-05-04_19-06',
    'grading_token_classification_2023-05-04_19-48',
    'grading_token_classification_2023-05-04_20-30',
    'grading_token_classification_2023-05-04_21-12',
    'grading_token_classification_2023-05-04_21-54',
    'grading_token_classification_2023-05-04_22-35',
    'grading_token_classification_2023-05-04_23-17',
    'grading_token_classification_2023-05-04_23-59',
    'grading_token_classification_2023-05-05_00-41',
    'grading_token_classification_2023-05-05_01-23',
    'grading_token_classification_2023-05-05_02-05',
    'grading_token_classification_2023-05-05_02-46',
    'grading_token_classification_2023-05-05_03-28',
    'grading_token_classification_2023-05-05_04-10',
    'grading_token_classification_2023-05-05_04-52',
    'grading_token_classification_2023-05-05_05-34',
    'grading_token_classification_2023-05-05_06-16',
    'grading_token_classification_2023-05-05_06-58',
    'grading_token_classification_2023-05-05_07-40',
    'grading_token_classification_2023-05-05_08-22',
    'grading_token_classification_2023-05-05_09-04',
    'grading_token_classification_2023-05-05_09-46',
    'grading_token_classification_2023-05-05_10-28',
    'grading_token_classification_2023-05-05_11-10',
    'grading_token_classification_2023-05-05_11-52',
    'grading_token_classification_2023-05-05_12-34',
    'grading_token_classification_2023-05-05_13-16',
    'grading_token_classification_2023-05-05_13-58',
    'grading_token_classification_2023-05-05_14-40',
    'grading_token_classification_2023-05-05_15-22',
    'grading_token_classification_2023-05-05_16-04',
    'grading_token_classification_2023-05-05_16-46',
    'grading_token_classification_2023-05-05_17-28',
    'grading_token_classification_2023-05-05_18-09',
    'grading_token_classification_2023-05-05_18-51',
    'grading_token_classification_2023-05-05_19-33',
    'grading_token_classification_2023-05-05_20-15',
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