# This script preprocesses the data for the justification cue detection training
import tokenizations
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from config import *
from dataset import *


# Helper functions
def _align_generate_labels_all_tokens(tokens_spacy, tokens_bert, l):
    a2b, b2a = tokenizations.get_alignments(tokens_spacy, tokens_bert)
    len_of_classification = len(tokens_bert)  # for CLS and end of seq
    label_ids = np.zeros((len_of_classification), dtype=np.int64)
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


def create_inputs(data, corpus, with_context=False):
    labels = []
    bert_tokens = []
    for d, l, c in zip(data, corpus):
        student_answer = d['student_answer']
        l = d['silver_label']
        tokenized_student_answer = tokenizer.encode(student_answer, add_special_tokens=False)
        tokens_spacy = [t.text for t in c]
        tokens_bert = [tokenizer.decode(t) for t in tokenized_student_answer]
        # Tokenize the input
        tokenized = tokenizer(student_answer, add_special_tokens=False)
        if with_context:
            context = d['reference_answer']
            length_stud_answer = len(tokenized[0]['input_ids'])
            tokenized = tokenizer(student_answer, context)
            input_ids = tokenized[0]['input_ids']
            attention_mask = np.ones(len(input_ids), dtype=np.int64)
            attention_mask[length_stud_answer:] = 0

        else:
            attention_mask = tokenized[0]['attention_mask']
        item = {
            'input_ids': tokenized[0]['input_ids'],
            'attention_mask':attention_mask ,
        }
        bert_tokens.append(item)
        # Generating the labels
        label_ids = _align_generate_labels_all_tokens(tokens_spacy, tokens_bert, l)
        labels.append(label_ids.tolist())

    return labels, bert_tokens


def tokenize(data):
    tokenized_data = []
    for i,d in data.iterrows():
        if d['lang'] == 'de':
            t = nlp_de(d['student_answer'])
        else:
            t= nlp(d['student_answer'])
        tokenized_data.append(t)
    return tokenized_data


#Load data
sep='\t'
X_train = pd.read_csv(PATH_DATA+'x_train.csv',  sep=sep)
X_dev = pd.read_csv(PATH_DATA+'x_dev.csv', sep=sep)

train_corpus = tokenize(X_train)
dev_corpus = tokenize(X_dev)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Preprocess data
bert_labels_train, bert_tokens_train = create_inputs(X_train, train_corpus)
bert_labels_dev, bert_tokens_dev = create_inputs(X_dev, dev_corpus)
training_dataset = CustomJustificationCueDataset(bert_tokens_train, bert_labels_train, X_train)
dev_dataset = CustomJustificationCueDataset(bert_tokens_dev, bert_labels_dev, X_dev)

#save data
torch.save(training_dataset, PATH_DATA+'training_dataset.pt')
torch.save(dev_dataset, PATH_DATA+'dev_dataset.pt')
