import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import GradingDataset
from grading_model import GradingModel
from preprocessor import GradingPreprocessor
import myutils as utils
from config import Config

TASK = 'token_classification'
CHECKPOINT_PATH = 'logs/grading_token_classification_classification_decision_tree/version_0/checkpoints/checkpoint-epoch=02-val_loss=0.91.ckpt'
CHECKPOINT_PATH_SYMBOLIC_MODELS = 'logs/grading_token_classification_classification_decision_tree/symbolic_models/epoch_1'
MODEL = 'distilbert-base-multilingual-cased'
CONTEXT = True
TEST_FILE = 'dev_dataset.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
label2idx = {'CORRECT': 0, 'PARTIAL_CORRECT': 1, 'INCORRECT': 2}
idx2label = {0: 'CORRECT', 1: 'PARTIAL_CORRECT', 2: 'INCORRECT'}
config = Config(task=TASK,
                        model=MODEL,
                        test_file=TEST_FILE,
                        checkpoint_path=CHECKPOINT_PATH,
                        context=CONTEXT,
                        device=DEVICE,
                        mode='classification',
                        grading_model='decision_tree',
                        batch_size=1,
                        )
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
symbolic_models = utils.load_symbolic_models(CHECKPOINT_PATH_SYMBOLIC_MODELS, rubrics)
model = GradingModel.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH, map_location=DEVICE, symbolic_models=symbolic_models)

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
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

predictions = []
for i, d in enumerate(test_loader):
    y_pred, scoring_vectors, justification_cues, justification_cue_texts = model.predict(d, i, return_reasoning=True)
    predictions.append({'y_pred': idx2label[y_pred.cpu().detach().numpy()[0].argmax()],
                        'scoring_vectors': scoring_vectors[0],
                        'justification_cues': justification_cues[0],
                        'justification_cue_spans': justification_cue_texts,
                        })

prediction_data = []
for d, p in zip(test_dataset, predictions):
    prediction_data.append({
        'id': d['question_id'],
        'text': d['student_answer'],
        'class': idx2label[d['class']],
        'score': utils.denormalize_score(d['score'], max_scores[d['question_id']]),
        'scoring_vectors': p['scoring_vectors'],
        'justification_cues': p['justification_cues'],
        'justification_cue_spans': p['justification_cue_spans'],
        'y_pred': p['y_pred'],
    })
utils.save_csv(prediction_data, config.PATH_RESULTS, 'final_prediction')
