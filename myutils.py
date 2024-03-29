import os
import time

import numpy as np
import pandas as pd
import skweak
from matplotlib import pyplot as plt
import json
from joblib import dump, load

def save_json(data, path, file_name, with_timestamp=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if with_timestamp:
        file_name + '_' + time.strftime("%Y_%m_%d")

    with open(path + '/' + file_name, 'w') as fout:
        json.dump(data, fout)
        print('saved', file_name)

def save_csv(data, path, file_name, with_timestamp=True, sep=','):
    if not os.path.exists(path):
        os.makedirs(path)
    if with_timestamp:
        file_name + '_' + time.strftime("%Y_%m_%d_%H_%M")
    data = pd.DataFrame(columns=data[0].keys(), data=data)
    data.to_csv(path + '/' + file_name + '.csv', index=False)
    print('saved', file_name + '.csv')


def load_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data


def flat_list(lst):
    x = [item for sublist in lst for item in sublist]
    return x


def load_rubrics(path):
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    rubrics = dict()
    for key in data:
        rubrics[key] = pd.DataFrame.from_dict(data[key])
    return rubrics


def prepare_rubrics(rubrics, config):
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


def tokenize_data(data, config):
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

def load_symbolic_models(path, rubrics):
    symbolic_models = {}
    for k in rubrics.keys():
        symbolic_models[k] = load(path + '/' + k + '.joblib')
    return symbolic_models

def get_experiment_name(constituents):
    EXPERIMENT_NAME = ''
    for i,c in enumerate(constituents):
        if i == len(constituents) - 1:
            EXPERIMENT_NAME += str(c).replace('/', '_')
        else:
            EXPERIMENT_NAME += str(c).replace('/', '_') + '_'
    return EXPERIMENT_NAME

def scaled_rounding(x, scale=0.125):
    # we round all values between 0 and 1 to the nearest interval of 0.125 (e.g. 0.125, 0.25, 0.375, ...)
    scaler = 1/scale
    # first we scale the values to the interval [0, scaler]
    x = x * scaler
    # then we round to the nearest integer
    x = np.round(x)
    # finally we scale back to the interval [0, 1]
    x = x / scaler
    return x

def normalize_score(score,  max_score):
    #min max normalization of the scores
    return score / max_score

def denormalize_score(score, max_score):
    return score * max_score