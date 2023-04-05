import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset import GradingDataset
from grading_model import GradingModel
from preprocessor import GradingPreprocessor
import myutils as utils
from config import Config

# Set the parameters here
TASK = 'span_prediction'
MODE = 'classification'
FOLDER = 'logs/grading_span_prediction_2023-04-04_04-27'
CHECKPOINT = 'checkpoint-epoch=03-val_loss=0.89.ckpt'
SYMBOLIC_MODELS_EPOCH = 'epoch_3'
SUB_FOLDER= '/version_0/checkpoints/'
TEST_FILE = 'dev_dataset.json'
MODEL = 'microsoft/mdeberta-v3-base'
LEARNING_STRATEGY = 'decision_tree'
CHECKPOINT_PATH = FOLDER + SUB_FOLDER  + CHECKPOINT
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

test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

trainer = Trainer(gpus=1, accelerator=DEVICE)
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

# more detailed metrics
print('Generate more detailed insights...')
y_true = [y[label] for y in prediction_data]
y_pred = [str(y['y_pred']) for y in prediction_data]

if MODE == 'classification':
    label_names = ['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT']
else:
    label_names = ['0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875', '1.0']

final_cm = {'confusion_matrix' : []}
final_cm['confusion_matrix'] = confusion_matrix(y_true=y_true,y_pred=y_pred, labels=label_names).tolist()
report = classification_report(y_true, y_pred, output_dict=True, labels=label_names)

# more detailed metrics per question
print('Generate more detailed insights per question...')

reports = {}
cms = {}
for qid in list(rubrics.keys()):
    y_true = [d[label] for d in prediction_data if d['question_id'] == qid]
    y_pred = [str(d['y_pred']) for d in prediction_data if d['question_id'] == qid]
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    reports[qid] = report
    cms[qid] = cm.tolist()

experiment_name = utils.get_experiment_name(['grading_final', LEARNING_STRATEGY, TASK, MODE])
os.mkdir(config.PATH_RESULTS + '/' + experiment_name) if os.path.exists(config.PATH_RESULTS + '/' + experiment_name) is False else None
save_path = config.PATH_RESULTS + '/' + experiment_name
utils.save_csv(prediction_data, save_path, 'final_prediction')
utils.save_json(reports, save_path, 'final_reports.json')
utils.save_json(cms, save_path, 'final_cms.json')
utils.save_json(final_cm, save_path, 'final_overall_cm.json')
utils.save_json(report, save_path, 'final_overall_report.json')
utils.save_json(metrics, save_path, 'final_metrics.json')