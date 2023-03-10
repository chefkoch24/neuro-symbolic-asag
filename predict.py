import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

import metrics
import myutils as utils
import config
import torch
from model import TokenClassificationModel
import numpy as np

NUM_SAMPLES = 20

# TOKEN CLASSIFICATION
test_file= 'dev_dataset_distilbert-base-multilingual-cased_context-False.json'
test_dataset = utils.load_json(config.PATH_DATA + '/' + test_file)

token_checkpoint_path ='logs/justification_cue_distilbert-base-multilingual-cased_context-False/version_7/checkpoints/checkpoint-epoch=04-val_loss=0.64.ckpt'

model_wo_context = TokenClassificationModel.load_from_checkpoint(token_checkpoint_path)

token_checkpoint_path = 'logs/justification_cue_distilbert-base-multilingual-cased_context-True/version_1/checkpoints/checkpoint-epoch=04-val_loss=0.64.ckpt'
model_w_context = TokenClassificationModel.load_from_checkpoint(token_checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
model_w_context.eval()
model_wo_context.eval()

with torch.no_grad():
    results = []
    for i, data in tqdm(enumerate(test_dataset)):
        # WITHOUT CONTEXT
        x = tokenizer(data['student_answer'], return_tensors='pt', return_token_type_ids=True, padding='max_length', max_length=512, truncation=True)
        logits = model_wo_context(x['input_ids'], x['attention_mask'])
        y_hat = torch.argmax(logits, dim=-1).detach().numpy()[0]
        input_ids = x['input_ids'].detach().numpy()[0]
        # remove CLS and SEP
        input_ids = input_ids[1:-1]
        y_hat=y_hat[1:-1]
        pred_spans = metrics.get_spans_from_labels(y_hat)
        pred_spans_wo_context = [tokenizer.decode(input_ids[s[0]:s[1]]) for s in pred_spans]
        # true spans
        true_labels = [np.argmax(l, axis=-1) for l in data['labels'] if l[1] != -100]
        spans = metrics.get_spans_from_labels(true_labels)
        true_spans =[tokenizer.decode(input_ids[s[0]:s[1]]) for s in spans]

        # WITH CONTEXT
        x = tokenizer(data['student_answer'], data['reference_answer'], return_tensors='pt',
                          return_token_type_ids=True, padding='max_length', max_length=512, truncation=True)
        logits = model_w_context(x['input_ids'], x['attention_mask'])
        y_hat = torch.argmax(logits, dim=-1).detach().numpy()[0]
        input_ids = x['input_ids'].detach().numpy()[0]
        len_student_answer = len(tokenizer(data['student_answer'])['input_ids'])
        # only keep student answer
        y_hat = y_hat[:len_student_answer]
        # remove CLS and SEP
        input_ids = input_ids[1:-1]
        y_hat = y_hat[1:-1]
        pred_spans = metrics.get_spans_from_labels(y_hat)
        pred_spans_w_context = [tokenizer.decode(input_ids[s[0]:s[1]]) for s in pred_spans]

        results.append({
                'question_id': data['question_id'],
                'student_answer': data['student_answer'],
                'true_spans': true_spans,
                'pred_spans_wo_context': pred_spans_wo_context,
                'pred_spans_w_context': pred_spans_w_context,
                'class': data['class']
            })
results = pd.DataFrame(columns=results[0].keys(), data=results)
result_file_name = 'predicitons_' + config.MODEL_NAME
results.to_csv('results/' + result_file_name + '.csv', index=False)