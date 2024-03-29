import numpy as np
import tokenizations
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
import myutils as utils

def align_generate_labels_all_tokens(tokens_spacy, tokens_bert, l):
    a2b, b2a = tokenizations.get_alignments(tokens_spacy, tokens_bert)
    len_of_classification = len(tokens_bert)  # for CLS and end of seq
    label_ids = np.zeros((len_of_classification))
    previous_label_idx = 0
    label_idx = -1
    for j, e in enumerate(b2a):
        if len(e) >= 1:  # Not special token
            label_idx = e[0]
            # if label_idx < len_of_classification:
            label_ids[j] = l[label_idx]
            previous_label_idx = label_idx
        else:
            label_ids[j] = l[previous_label_idx]
    # label_ids[len_of_classification:] = -100
    return label_ids


def create_aligned_labels(data):
    aligned_data = []
    for d in tqdm(data):
        student_answer = d['student_answer']
        labels = d['silver_labels']
        nlp = config.nlp_de if d['lang'] == 'de' else config.nlp
        tokens_spacy = [t.text for t in nlp(student_answer)]
        # Tokenize the input to generate alignment
        tokenized = tokenizer(student_answer, add_special_tokens=False)
        tokens_bert = [tokenizer.decode(t) for t in tokenized['input_ids']] # used for alingment
        aligned_labels = align_generate_labels_all_tokens(tokens_spacy, tokens_bert, labels).tolist()
        # get the spans from the aligned labels
        d['aligned_labels'] = aligned_labels
        aligned_data.append(d)
    return aligned_data

train_file = 'aggregated/training/training_ws_hmm_1_all_lfs.json'
dev_file = 'aggregated/dev/dev_ws_hmm_1_all_lfs.json'

for model in ['distilbert-base-multilingual-cased', 'microsoft/mdeberta-v3-base']:
        config = Config(train_file=train_file,
                        dev_file=dev_file,
                        model=model,)
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)
        rubrics = utils.load_rubrics(config.PATH_RUBRIC)
        DATASET_NAME = utils.get_experiment_name(['aligned_labels', config.MODEL_NAME]) + '.json'

        # load data
        train_data = utils.load_json(config.PATH_DATA + '/' + config.TRAIN_FILE)
        dev_data = utils.load_json(config.PATH_DATA + '/' + config.DEV_FILE)
        training_dataset = create_aligned_labels(train_data)
        dev_dataset = create_aligned_labels(dev_data)
        utils.save_json(training_dataset, config.PATH_DATA + '/', 'training_' + DATASET_NAME)
        utils.save_json(dev_dataset, config.PATH_DATA + '/', 'dev_'+DATASET_NAME)