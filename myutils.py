import os

import numpy as np
import pandas as pd
import skweak
import tokenizations
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import torch
import json

import config


def save_json(data, path, file_name):
    directory = path
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(path + '/' + file_name, 'w') as fout:
        json.dump(data, fout)
        print('saved', file_name)


def load_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data


def flat_list(lst):
    x = [item for sublist in lst for item in sublist]
    return x


def save_to_csv(X_train, X_dev, y_train, y_dev, path):
    sep = "\t"
    directory = path

    if not os.path.exists(directory):
        os.mkdir(directory)

    save_path = path + '/'
    X_train.to_csv(save_path + "x_train.tsv", sep=sep)
    X_dev.to_csv(save_path + "x_dev.tsv", sep=sep)
    pd.DataFrame(data=y_train).to_csv(save_path + "y_train.tsv", sep=sep)
    pd.DataFrame(data=y_dev).to_csv(save_path + "y_dev.tsv", sep=sep)
    print('successfully saved')


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


def load_rubrics(path):
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    rubrics = dict()
    for key in data:
        rubrics[key] = pd.DataFrame.from_dict(data[key])
    return rubrics


def prepare_rubrics(rubrics):
    german_question_ids = [str(i) for i in range(1, 10)]
    for key in rubrics:
        rubric = rubrics[key]
        tokenezied_elements = []
        for i, r in rubric.iterrows():
            key_element = r['key_element']
            if key in german_question_ids:
                tokenized = config.nlp_de(key_element)
            else:
                tokenized = config.nlp(key_element)
            tokenezied_elements.append(tokenized)
        rubric['tokenized'] = tokenezied_elements
        rubrics[key] = rubric
    return rubrics


def save_annotated_corpus(annotated_docs, path):
    print(len(annotated_docs))
    for doc in annotated_docs:
        doc.ents = doc.spans["hmm"]
    skweak.utils.docbin_writer(annotated_docs, path)


def tokenize_data(data):
    tokenized = []
    for i, d in data.iterrows():
        if d['lang'] == 'en':
            d = config.nlp(d['student_answer'])
        elif d['lang'] == 'de':
            d = config.nlp_de(d['student_answer'])
        tokenized.append(d)
    data['tokenized'] = tokenized
    return data


def create_labels_probability_distribution(labels):
    targets = []
    for l in labels:
        if l > -100:
            targets.append([1 - l, l])
        else:
            targets.append([l, l])
    return targets


def plot_hist(stats, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], title=""):
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Frequency')
    plt.hist([stats['CORRECT'], stats['PARTIAL_CORRECT'], stats['INCORRECT']], bins=bins, label=list(stats.keys()))
    plt.legend()
    plt.show()

