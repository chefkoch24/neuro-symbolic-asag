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
import utils_preprocessing
warnings.filterwarnings("ignore")

#Set seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

def get_rubric_elements(spans, input_ids, qid):
    decoded = [tokenizer.decode(i, skip_special_tokens=True) for i in input_ids]
    rubric = rubrics[qid]
    rubric_elements = []
    for s in spans:
        span_text = decoded[s[0]:s[1]]
        sim = para_detector.detect_score_key_elements(''.join(span_text), rubric)
        max_index = np.argmax(sim)
        rubric_element = rubric['key_element'][max_index]
        rubric_elements.append(rubric_element)
    return rubric_elements


def create_inputs(data):
    model_inputs = []
    for d in tqdm(data):
        student_answer = d['student_answer']
        labels = d['silver_labels']
        q_id = d['question_id']
        nlp = config.nlp_de if d['lang'] == 'de' else config.nlp
        tokens_spacy = [t.text for t in nlp(student_answer)]
        # Tokenize the input to generate alignment
        tokenized = tokenizer(student_answer, add_special_tokens=False, return_offsets_mapping=True)
        tokens_bert = [tokenizer.decode(t) for t in tokenized['input_ids']] # used for alingment
        aligned_labels = utils_preprocessing.align_generate_labels_all_tokens(tokens_spacy, tokens_bert, labels).tolist()
        # get the offset mappings from the tokenized input without special tokens
        offset_mapping = tokenized['offset_mapping']
        # get the spans from the aligned labels
        hard_labels = metrics.silver2target(aligned_labels)
        spans = metrics.get_spans_from_labels(hard_labels)
        rubric_elements = get_rubric_elements(spans, tokenized['input_ids'], q_id)
        # generate final input for every span individually
        for span, re in zip(spans, rubric_elements):
            tokenized = tokenizer(student_answer, re, max_length=config.MAX_LEN, truncation=True, padding='max_length')
            model_input = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'start_positions': [span[0]],
                'end_positions': [span[1]],
                'question_id': q_id,
                'rubric_element': re,
                'class': d['label']
            }
            model_inputs.append(model_input)
    return model_inputs

#Loading
train_data = utils.load_json(config.PATH_DATA + '/' + config.ANNOTATED_TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.ANNOTATED_DEV_FILE)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
para_detector = BertScorer()

training_dataset = create_inputs(train_data)
dev_dataset = create_inputs(dev_data)

#save data
DATASET_NAME = 'dataset_span_prediction_' + config.MODEL_NAME + '.json'
utils.save_json(training_dataset, config.PATH_DATA + '/', 'training_' + DATASET_NAME)
utils.save_json(dev_dataset, config.PATH_DATA + '/', 'dev_'+DATASET_NAME)

