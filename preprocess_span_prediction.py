import argparse
import numpy as np
import torch
import metrics
import myutils as utils
import config
from transformers import AutoTokenizer
import warnings
from paraphrase_scorer import ParaphraseScorerSBERT, BertScorer
from tqdm import tqdm
warnings.filterwarnings("ignore")

#Set seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

def get_rubric_elements(spans, input_ids, qid):
    rubric = rubrics[qid]
    rubric_elements = []
    sims = []
    for s in spans:
        span_text = tokenizer.decode(input_ids[s[0]:s[1]])
        sim = para_detector.detect_score_key_elements(span_text, rubric)
        max_index = np.argmax(sim)
        rubric_element = rubric['key_element'][max_index]
        rubric_elements.append(rubric_element)
        sims.append(np.max(sim))
    return rubric_elements, sims


def pre_process(data):
    model_inputs = []
    for d in tqdm(data):
        student_answer = d['student_answer']
        aligned_labels = d['aligned_labels']
        q_id = d['question_id']
        # Tokenize the input to generate alignment
        tokenized = tokenizer(student_answer, add_special_tokens=False, return_offsets_mapping=True)
        # get the spans from the aligned labels
        hard_labels = metrics.silver2target(aligned_labels)
        spans = metrics.get_spans_from_labels(hard_labels)
        rubric_elements, rubric_sims = get_rubric_elements(spans, tokenized['input_ids'], q_id)
        # generate final input for every span individually
        for span, re, sim in zip(spans, rubric_elements, rubric_sims):
            tokenized = tokenizer(student_answer, re, max_length=config.MAX_LEN, truncation=True, padding='max_length', return_token_type_ids=True)
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

#Loading
train_data = utils.load_json(config.PATH_DATA + '/' + config.ALIGNED_TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.ALIGNED_DEV_FILE)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
para_detector = BertScorer()

training_dataset = pre_process(train_data)
dev_dataset = pre_process(dev_data)

#save data
DATASET_NAME = 'dataset_span_prediction_' + config.MODEL_NAME + '.json'
utils.save_json(training_dataset, config.PATH_DATA + '/', 'training_' + DATASET_NAME)
utils.save_json(dev_dataset, config.PATH_DATA + '/', 'dev_'+DATASET_NAME)
