import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import GradingDataset, CustomBatchSampler
from grading_model import GradingModel, Summation
from preprocessor import GradingPreprocessor
import myutils as utils
from config import Config

# Uncomment and set the parameters here
LEARNING_STRATEGY = 'decision_tree'
# Config 1: Token Classification DT
#TASK = 'token_classification'
#MODE = 'classification'
#FOLDER = 'logs/grading_token_classification_2023-04-05_12-04' # token
#CHECKPOINT = 'checkpoint-epoch=02-val_loss=0.86.ckpt' #classification
#SYMBOLIC_MODELS_EPOCH = 'epoch_2'

# Config 2: Token Regression DT
TASK = 'token_classification'
MODE = 'regression'
FOLDER = 'logs/grading_token_classification_2023-04-05_04-02'
CHECKPOINT = 'checkpoint-epoch=02-val_loss=0.08.ckpt' #regression
SYMBOLIC_MODELS_EPOCH = 'epoch_2'

# Config 3: Span Classification DT
#TASK = 'span_prediction'
#MODE = 'classification'
#FOLDER = 'logs/grading_span_prediction_2023-04-06_20-10' #classification
#CHECKPOINT = 'checkpoint-epoch=03-val_loss=0.87.ckpt' #classification
#SYMBOLIC_MODELS_EPOCH = 'epoch_3'

#Config 4: Span Regression DT
#TASK = 'span_prediction'
#MODE = 'regression'
#FOLDER = 'logs/grading_span_prediction_2023-04-06_16-19' #regression
#CHECKPOINT = 'checkpoint-epoch=01-val_loss=0.09.ckpt' #regression
#SYMBOLIC_MODELS_EPOCH = 'epoch_1'

#LEARNING_STRATEGY = 'summation'
# Config 5: Token Classification Summation
#TASK = 'token_classification'
#MODE = 'classification'
#FOLDER = 'logs/grading_token_classification_2023-04-07_01-43'
#CHECKPOINT = 'checkpoint-epoch=01-val_loss=1.42.ckpt'
#SYMBOLIC_MODELS_EPOCH = 'epoch_1'

# Config 6: Token Regression Summation
#TASK = 'token_classification'
#MODE = 'regression'
#FOLDER ='logs/grading_token_classification_2023-04-07_00-35'
#CHECKPOINT = 'checkpoint-epoch=01-val_loss=0.61.ckpt'
#SYMBOLIC_MODELS_EPOCH = 'epoch_1'

# Config 7: Span Classification Summation
#TASK = 'span_prediction'
#MODE = 'classification'
#FOLDER = 'logs/grading_span_prediction_2023-04-06_22-06'
#CHECKPOINT = 'checkpoint-epoch=01-val_loss=1.31.ckpt'
#SYMBOLIC_MODELS_EPOCH = 'epoch_1'

# Config 8: Span Regression Summation
#TASK = 'span_prediction'
#MODE = 'regression'
#FOLDER = 'logs/grading_span_prediction_2023-04-06_18-14'
#CHECKPOINT = 'checkpoint-epoch=01-val_loss=0.50.ckpt'
#SYMBOLIC_MODELS_EPOCH = 'epoch_1'

# Shared settings
SUB_FOLDER= '/version_0/checkpoints/'

TEST_FILE = 'test_dataset.json'
MODEL = 'microsoft/mdeberta-v3-base'
CHECKPOINT_PATH = FOLDER + SUB_FOLDER + CHECKPOINT
CHECKPOINT_PATH_SYMBOLIC_MODELS = FOLDER + '/symbolic_models/' + SYMBOLIC_MODELS_EPOCH
CONTEXT = True if TASK == 'token_classification' else None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
label2idx = {'CORRECT': 0, 'PARTIAL_CORRECT': 1, 'INCORRECT': 2}
idx2label = {0: 'CORRECT', 1: 'PARTIAL_CORRECT', 2: 'INCORRECT'}
label = 'class' if MODE == 'classification' else 'score'
config = Config(task=TASK,
                model=MODEL,
                test_file=TEST_FILE,
                checkpoint_path=CHECKPOINT_PATH,
                context=CONTEXT,
                device=DEVICE,
                mode=MODE,
                grading_model=LEARNING_STRATEGY,
                batch_size=4, # for compatibility baseline
                )
seed_everything(config.SEED)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
symbolic_models = utils.load_symbolic_models(CHECKPOINT_PATH_SYMBOLIC_MODELS, rubrics)
model = GradingModel.load_from_checkpoint(
    checkpoint_path=CHECKPOINT_PATH,
    map_location=DEVICE,
    symbolic_models=symbolic_models)

preprocessor = GradingPreprocessor(config.MODEL_NAME, with_context=config.CONTEXT,
                                   rubrics=rubrics)
test_data = utils.load_json(config.PATH_DATA + '/' + config.TEST_FILE)
test_dataset = preprocessor.preprocess(test_data)
test_dataset = GradingDataset(test_dataset)
max_scores = {}
for k in rubrics.keys():
    scores = [d['score'] for d in test_dataset if d['question_id'] == k]
    if scores != []:
        max_scores[k] = np.max(scores)
    else:
        max_scores[k] = 0

test_loader = DataLoader(test_dataset, batch_sampler=CustomBatchSampler(test_dataset, config.BATCH_SIZE, seed=config.SEED))

trainer = Trainer(gpus=1,
                  accelerator=DEVICE,
                  deterministic=True,
                  )
# report the standard metrics for the task on the test set
metrics = trainer.test(model, test_loader)

# new test load with batch size 1, because we want to get the reasoning
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Do and save the predictions
print('Generate the predictions...')
predictions = []
for i, d in tqdm(enumerate(test_loader)):
    y_pred, scoring_vectors, justification_cues, justification_cue_texts = model.predict(d, i, return_reasoning=True)
    if MODE == 'classification':
        y_pred = idx2label[y_pred.cpu().detach().numpy()[0].argmax()]
    else:
        y_pred = utils.denormalize_score(y_pred.cpu().detach().numpy()[0], max_scores[d['question_id'][0]])
    predictions.append({'y_pred': y_pred,
                        'scoring_vectors': scoring_vectors[0],
                        'justification_cues': justification_cues[0],
                        'justification_cue_spans': justification_cue_texts,
                        })
# add all the meta data
print('Create the prediction data...')
prediction_data = []
for d, p in tqdm(zip(test_dataset, predictions)):
    prediction_data.append({
        'question_id': d['question_id'],
        'text': d['student_answer'],
        'class': idx2label[d['class']],
        'score': utils.denormalize_score(d['score'], max_scores[d['question_id']]),
        'scoring_vectors': p['scoring_vectors'],
        'justification_cues': p['justification_cues'],
        'justification_cue_spans': p['justification_cue_spans'],
        'y_pred': p['y_pred'],
    })

experiment_name = utils.get_experiment_name(['grading_final', LEARNING_STRATEGY, TASK, MODE])

if TEST_FILE == 'dev_dataset.json':
   experiment_name = 'dev_' + experiment_name
os.mkdir(config.PATH_RESULTS + '/' + experiment_name) if os.path.exists(config.PATH_RESULTS + '/' + experiment_name) is False else None
save_path = config.PATH_RESULTS + '/' + experiment_name
utils.save_csv(prediction_data, save_path, 'final_prediction')
