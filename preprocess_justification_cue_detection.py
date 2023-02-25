# This script preprocesses the data for the justification cue detection training
import argparse
import tokenizations
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import config
import myutils as utils
from dataset import JustificationCueDataset
import torch
import utils_preprocessing

# Helper functions
def create_inputs(data, with_context=False):
    model_inputs = []
    for d in data:
        student_answer = d['student_answer']
        l = d['silver_labels']
        nlp = config.nlp_de if d['lang'] == 'de' else config.nlp
        tokens_spacy = [t.text for t in nlp(student_answer)]
        # Tokenize the input
        tokenized = tokenizer(student_answer, add_special_tokens=False)
        tokens_bert = [tokenizer.decode(t) for t in tokenized['input_ids']]
        if with_context:
            context = d['reference_answer']
            tokenized = tokenizer(student_answer, context, max_length=config.MAX_LEN, truncation=True, padding='max_length')

        else:
            tokenized = tokenizer(student_answer, max_length=config.MAX_LEN, truncation=True, padding='max_length')
        # Generating the labels
        aligned_labels = utils_preprocessing.align_generate_labels_all_tokens(tokens_spacy, tokens_bert, l)
        pad_len = config.MAX_LEN - len(aligned_labels)
        labels = [-100]+aligned_labels.tolist()
        # Adding other model inputs
        model_input = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': utils.create_labels_probability_distribution(torch.nn.functional.pad(torch.tensor(labels), pad=(0, pad_len), mode='constant', value=-100).detach().numpy().tolist()),
            'class': d['label'],
            'question_id': d['question_id']
        }
        model_inputs.append(model_input)

    return model_inputs


#Loading
train_data = utils.load_json(config.PATH_DATA + '/' + config.TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.DEV_FILE)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

# Preprocess data
training_dataset = create_inputs(train_data, with_context=config.context)
dev_dataset = create_inputs(dev_data, with_context=config.context)

#save data
DATASET_NAME = 'dataset'+ '_' + config.MODEL_NAME + '_context-' + str(config.context) + '.json'
utils.save_json(training_dataset, config.PATH_DATA + '/', 'training_' + DATASET_NAME)
utils.save_json(dev_dataset, config.PATH_DATA + '/', 'dev_'+DATASET_NAME)
