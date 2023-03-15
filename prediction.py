import io
import logging
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import metrics
import myutils as utils
from dataset import *
from model import SpanPredictionModel, TokenClassificationModel
from preprocessor import *

def save_results(results, config):
    results = pd.DataFrame(columns=results[0].keys(), data=results)
    result_file_name = 'predicitons_' + config.MODEL_NAME.replace('/', '_')
    results.to_csv('results/' + result_file_name + '.csv', index=False)


class PredictToken:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = TokenClassificationModel.load_from_checkpoint(self.config.PATH_CHECKPOINT)
        self.with_context = self.config.CONTEXT
        self.test_dataset = utils.load_json(self.config.PATH_DATA + '/' + self.config.TEST_FILE)
        preprocessor = PreprocessorTokenClassification(self.config.MODEL_NAME, with_context=self.config.CONTEXT)
        self.test_dataset = preprocessor.preprocess(self.test_dataset)
        self.test_dataset = JustificationCueDataset(self.test_dataset)

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            results = []
            for i, data in tqdm(enumerate(self.test_dataset)):
                logits = self.model.forward(input_ids=data['input_ids'].unsqueeze(1), attention_mask=data['attention_mask'].unsqueeze(1))
                attention_mask = data['attention_mask'].cpu().numpy() == 1
                token_type_ids = data['token_type_ids'].cpu().numpy() == 0
                valid_indices = attention_mask & token_type_ids
                y_hat = torch.argmax(logits[valid_indices], dim=-1).detach().numpy()
                input_ids = data['input_ids'].cpu().numpy()[valid_indices]
                # remove CLS and SEP
                input_ids = input_ids[1:-1]
                y_hat = utils.flat_list(y_hat[1:-1])
                pred_spans = metrics.get_spans_from_labels(y_hat)
                pred_spans = [self.tokenizer.decode(input_ids[s[0]:s[1]]) for s in pred_spans]
                # true spans
                true_labels = [torch.argmax(l, axis=-1).item() for l in data['labels'] if l[1] != -100]
                spans = metrics.get_spans_from_labels(true_labels)
                true_spans = [self.tokenizer.decode(input_ids[s[0]:s[1]]) for s in spans]
                results.append({
                    'question_id': data['question_id'],
                    'student_answer': data['student_answer'],
                    'true_spans': true_spans,
                    'pred_spans': pred_spans,
                    'class': data['class']
                })
        save_results(results, self.config)


class PredictSpan:
    def __init__(self, config):
        self.config = config
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = SpanPredictionModel.load_from_checkpoint(config.PATH_CHECKPOINT)
        self.test_dataset = utils.load_json(self.config.PATH_DATA + '/' + self.config.TEST_FILE)
        # TokenClassification is used intentional as it is faster and we want to predict over all rubric items
        preprocessor = PreprocessorTokenClassification(self.config.MODEL_NAME)
        self.test_dataset = preprocessor.preprocess(self.test_dataset)
        self.test_dataset = JustificationCueDataset(self.test_dataset)
        self.rubrics = utils.load_rubrics(config.PATH_RUBRIC)

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            results = []
            for i, data in tqdm(enumerate(self.test_dataset)):
                q_id = data['question_id']
                for re in self.rubrics[q_id]['key_element']:
                    x = self.tokenizer(re, data['student_answer'],return_tensors='pt', return_token_type_ids=True,
                                   padding='max_length',
                                   max_length=512, truncation=True)
                    start_logits, end_logits = self.model.forward(x['input_ids'], x['attention_mask'])

                    mask = (x['token_type_ids'] == 1) & (x['attention_mask'] == 1)
                    start_logits_masked = start_logits.masked_fill(~mask, float('-inf'))
                    end_logits_masked = end_logits.masked_fill(~mask, float('-inf'))

                    start = start_logits_masked.argmax(dim=-1).item()
                    end = end_logits_masked.argmax(dim=-1).item()
                    # remove CLS and SEP
                    input_ids = x['input_ids'][0][1:-1]
                    span = ''
                    if start < end:
                        span = self.tokenizer.decode(input_ids)
                    results.append({
                        'question_id': data['question_id'],
                        'student_answer': data['student_answer'],
                        'prediction': span,
                        'rubric_element': re,
                        'class': data['class']
                    })
        save_results(results, self.config)