from abc import abstractmethod
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import metrics
import myutils as utils

class Preprocessor:
    def __init__(self, tokenizer:str, max_len:int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len

    @abstractmethod
    def preprocess(self, data):
        pass

class PreprocessorTokenClassification(Preprocessor):

    def __init__(self, tokenizer, max_len=512, with_context=False):
        super().__init__(tokenizer, max_len)
        self.with_context = with_context

    def preprocess(self, data):
        model_inputs = []
        for d in tqdm(data):
            student_answer = d['student_answer']
            # Tokenize the input
            if self.with_context:
                context = d['reference_answer']
                tokenized = self.tokenizer(student_answer, context, max_length=self.max_len, truncation=True, padding='max_length', return_token_type_ids=True)

            else:
                tokenized = self.tokenizer(student_answer, max_length=self.max_len, truncation=True, padding='max_length', return_token_type_ids=True)
            # Generating the labels
            aligned_labels = d['aligned_labels']
            pad_len = self.max_len - len(aligned_labels) -2
            labels = [-100]+aligned_labels + [-100]
            # Adding other model inputs
            model_input = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'token_type_ids': tokenized['token_type_ids'],
                'labels': utils.create_labels_probability_distribution(torch.nn.functional.pad(torch.tensor(labels), pad=(0, pad_len), mode='constant', value=-100).detach().numpy().tolist()),
                'class': d['label'],
                'question_id': d['question_id'],
                'student_answer': d['student_answer'],
                'reference_answer': d['reference_answer'],
            }
            model_inputs.append(model_input)

        return model_inputs

class PreprocessorSpanPrediction(Preprocessor):
    def __init__(self, tokenizer, max_len=512, scorer=None, rubrics=None):
        super().__init__(tokenizer, max_len)
        self.scorer = scorer
        self.rubrics = rubrics

    def get_rubric_elements(self, spans, input_ids, qid):
        rubric = self.rubrics[qid]
        rubric_elements = []
        sims = []
        for s in spans:
            span_text = self.tokenizer.decode(input_ids[s[0]:s[1]])
            sim = self.scorer.detect_score_key_elements(span_text, rubric)
            max_index = np.argmax(sim)
            rubric_element = rubric['key_element'][max_index]
            rubric_elements.append(rubric_element)
            sims.append(np.max(sim))
        return rubric_elements, sims

    def preprocess(self, data):
        model_inputs = []
        for d in tqdm(data):
            student_answer = d['student_answer']
            aligned_labels = d['aligned_labels']
            q_id = d['question_id']
            # Tokenize the input to generate alignment
            tokenized = self.tokenizer(student_answer, add_special_tokens=False, return_offsets_mapping=True)
            # get the spans from the aligned labels
            hard_labels = metrics.silver2target(aligned_labels)
            spans = metrics.get_spans_from_labels(hard_labels)
            rubric_elements, rubric_sims = self.get_rubric_elements(spans, tokenized['input_ids'], q_id)
            # generate final input for every span individually
            for span, re, sim in zip(spans, rubric_elements, rubric_sims):
                tokenized = self.tokenizer(re, student_answer, max_length=self.max_len, truncation=True,
                                      padding='max_length', return_token_type_ids=True)
                model_input = {
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'token_type_ids': tokenized['token_type_ids'],
                    'start_positions': [span[0]],
                    'end_positions': [span[1]],
                    'question_id': q_id,
                    'rubric_element': re,
                    'class': d['label'],
                    'similarity': sim,
                    'student_answer': student_answer
                }
                model_inputs.append(model_input)
        return model_inputs

class GradingPreprocessorTokenClassification(Preprocessor):
    def __init__(self, tokenizer, max_len=512, with_context=False, rubrics=None, class2idx={'CORRECT': 0, 'PARTIAL_CORRECT': 1, 'INCORRECT': 2}):
        super().__init__(tokenizer, max_len)
        self.rubrics = rubrics
        self.with_context = with_context
        self.class2idx = class2idx

    def preprocess(self, data):
        model_inputs = []
        max_scores = {}
        for k in self.rubrics.keys():
            scores = [d['score'] for d in data if d['question_id'] == k]
            if scores != []:
                max_scores[k] = np.max(scores)
            else:
                max_scores[k] = 0
        for d in data:
            q_id = d['question_id']
            if self.with_context:
                tokenized = self.tokenizer(d['student_answer'], d['reference_answer'], truncation=True, padding='max_length',
                                      max_length=self.max_len, return_tensors='pt', return_token_type_ids=True)
            else:
                tokenized = self.tokenizer(d['student_answer'], truncation=True, padding='max_length', max_length=self.max_len,
                                      return_tensors='pt', return_token_type_ids=True)
            d['input_ids'] = tokenized['input_ids']
            d['attention_mask'] = tokenized['attention_mask']
            d['class'] = self.class2idx[d['label']]
            d['token_type_ids'] = tokenized['token_type_ids']
            d['score'] = normalize_score(d['score'], max_scores[q_id])
            model_inputs.append({
                'input_ids': d['input_ids'],
                'attention_mask': d['attention_mask'],
                'token_type_ids': d['token_type_ids'],
                'question_id': d['question_id'],
                'student_answer': d['student_answer'],
                'reference_answer': d['reference_answer'],
                'class': d['class'],
                'score': d['score']
            })
        return model_inputs

class GradingPreprocessorSpanPrediction(Preprocessor):
    def __int__(self, tokenizer, max_len=512, rubrics=None, class2idx={'CORRECT': 0, 'PARTIAL_CORRECT': 1, 'INCORRECT': 2}):
        super().__init__(tokenizer, max_len)
        self.rubrics = rubrics
        self.class2idx = class2idx

    def preprocess(self, data):
        model_inputs = []
        max_scores = {}
        for k in self.rubrics.keys():
            scores = [d['score'] for d in data if d['question_id'] == k]
            if scores != []:
                max_scores[k] = np.max(scores)
            else:
                max_scores[k] = 0
        for d in data:
            q_id = d['question_id']
            for re in self.rubrics[q_id]['key_element']:
                tokenized = self.tokenizer(re, d['student_answer'], truncation=True,
                                           padding='max_length',
                                           max_length=self.max_len, return_tensors='pt', return_token_type_ids=True)
                d['input_ids'] = tokenized['input_ids']
                d['attention_mask'] = tokenized['attention_mask']
                d['class'] = self.class2idx[d['label']]
                d['token_type_ids'] = tokenized['token_type_ids']
                d['score'] = normalize_score(d['score'], max_scores[q_id])
                model_inputs.append(
                    {
                        'input_ids': d['input_ids'],
                        'attention_mask': d['attention_mask'],
                        'token_type_ids': d['token_type_ids'],
                        'question_id': q_id,
                        'rubric_element': re,
                        'class': d['class'],
                        'score': d['score'],
                    }
            )
        return model_inputs

def normalize_score(score,  max_score):
    #min max normalization of the scores
    return score / max_score