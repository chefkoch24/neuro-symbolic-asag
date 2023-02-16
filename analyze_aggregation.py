# Settings

exclude = []
# exclude all hard labeling functions
#exclude = ['LF_lemma_match_without_stopwords', 'LF_pos_match_without_stopwords', 'LF_dep_match', 'LF_pos_match', 'LF_tag_match', 'LF_bleu_candidates','LF_edit_distance', 'LF_jaccard_similarity']
# exclude functions with low performance
#exclude = ['LF_pos_match', 'LF_tag_match', 'LF_dep_match','LF_pos_match_without_stopwords','LF_edit_distance', 'LF_jaccard_similarity']
TOKENIZER_NAME = 'distilroberta-base'
th = 0.49
GLOBAL_NORMALIZE = False
average_outliers = False


# Generate filename
results_name = 'results_global_normalize' if GLOBAL_NORMALIZE else 'results'
results_name += '_average_outliers' if average_outliers else ''

#Imports
import json
import spacy
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from spacy import displacy
import config
import matplotlib.pyplot as plt
import skweak
import myutils as utils

# Functions
def global_normalize(data, average_outliers=False):
    # finding min and max value
    min_value = 100
    max_value = -100
    max_vals = []
    min_vals = []
    for d in data:
        v_min = np.min(d)
        v_max = np.max(d)
        if v_min < min_value:
            min_value = v_min
            min_vals.append(min_value)
        if v_max > max_value:
            max_value = v_max
            max_vals.append(max_value)
    #print('max', max_vals, 'min', min_vals)
    #min_value = min_vals[0]
    #max_value = max_vals[0]
    if average_outliers:
        min_value = np.average(min_vals)
        max_value = np.average(max_vals)
    range_val = max_value - min_value
    data_norm = [[(l - min_value) / range_val for l in d] for d in data]
    return data_norm

def read_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def extract_annotations(annotated_data, exclude_LFs=[]):
    labels = []
    for a in annotated_data:
        soft_labels = []
        for k,v in a['labeling_functions'].items():
            if k not in exclude_LFs:
                soft_labels.append(v)
        labels.append(soft_labels)
    return labels

def normalize(sequence):
    min_value = min(sequence)
    max_value = max(sequence)
    range_value = max_value - min_value
    if range_value == 0:
        range_value = 0.001
    normalized_sequence = [(x - min_value) / range_value for x in sequence]
    return normalized_sequence


def silver2target(data, th=0.5):
    targets = []
    for l in data:
        if l >= th:
            targets.append(1)
        else:
            targets.append(0)
    return targets

def plot_hist(stats, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], title=""):
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Frequency')
    plt.hist([stats['CORRECT'],stats['PARTIAL_CORRECT'],stats['INCORRECT']], bins=bins, label=list(stats.keys()))
    plt.legend()
    plt.show()

def mean_nonzero(col):
    return np.mean([c for c in col if c!=0])

def aggregate_soft_labels(soft_labels, mode:str):
    if mode == 'average':
        return np.average(soft_labels, axis=0)
    elif mode == 'max':
        return np.max(soft_labels, axis=0)
    elif mode == 'average_nonzero':
        return np.apply_along_axis(mean_nonzero, axis=0, arr=soft_labels)
    elif mode == 'sum':
        return np.sum(soft_labels, axis=0)

# Those functions should work on none batched data
def rubric_length(labels):
    lens = []
    prev = -1
    cur = 0
    for i,l in enumerate(labels):
        if i == len(labels)-1:
            if l == 1:
                cur +=1
                lens.append(cur) # last rubric
            elif prev == 1:
                lens.append(cur)
        elif l == 1:
            cur +=1
        elif l == 0 and prev == 1:
            lens.append(cur) # within rubric
            cur = 0
        prev = l
    return lens


def relation(y):
    # relation of how many tokens are class 1 compared to all tokens
    class_1 = 0
    class_0 = 0
    for l in y:
            if l == 1:
                class_1 +=1
            elif l == 0:
                class_0 +=1
    return class_1/(class_0+class_1)

def plot_bars(classes, data, title):
    plt.bar(classes, data)
    plt.title(title)
    plt.show()

# Loading
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
nlp = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")
german_question_ids = [str(i) for i in range(1,9)]
with open(config.PATH_DATA + '/' + 'rubrics.json', 'r') as f:
    data = json.load(f)
rubrics = dict()
for key in data:
    rubrics[key] = pd.DataFrame.from_dict(data[key])
annotated_train_data = read_from_json(config.PATH_DATA + '/' +'train-soft.json')
annotated_dev_data = read_from_json(config.PATH_DATA + '/' + 'dev-soft.json')

# Preparing data
annotations_train = extract_annotations(annotated_train_data, exclude_LFs=exclude)
annotations_dev = extract_annotations(annotated_train_data, exclude_LFs=exclude)

#Creating results
results = []
disable_lang_filter = False

for mode in ['average', 'max', 'average_nonzero', 'sum']:
    print(mode.upper())
    silver_labels = []
    for data in annotations_train:
        y = aggregate_soft_labels(data, mode)
            #print(y)
        if GLOBAL_NORMALIZE == False:
            y = normalize(y)
        silver_labels.append(y)
    if GLOBAL_NORMALIZE == True:
        silver_labels = global_normalize(silver_labels, average_outliers=average_outliers)


    for language in ['de', 'en', 'combined']:
        relations_correct, relations_partial, relations_incorrect = [], [], []
        len_rubrics_correct, len_rubrics_partial, len_rubrics_incorrect = [], [], []
        len_rubrics_average_correct, len_rubrics_average_partial, len_rubrics_average_incorrect = [], [], []
        if language == 'combined':
            disable_lang_filter = True
        for an, labels in zip(annotated_train_data, silver_labels):
            lang = an['lang']
            c = an['label']
            if lang == language or disable_lang_filter:
                true_labels = silver2target(labels, th=th)
                rel = relation(true_labels)
                len_rubrics = rubric_length(true_labels)
                if len(len_rubrics) > 0:
                    len_rubrics_average = np.average(len_rubrics)
                else:
                    len_rubrics_average = 0
                if c == 'CORRECT':
                    relations_correct.append(rel)
                    len_rubrics_correct.append(len(len_rubrics))
                    len_rubrics_average_correct.append(len_rubrics_average)
                elif c == 'PARTIAL_CORRECT':
                    relations_partial.append(rel)
                    len_rubrics_partial.append(len(len_rubrics))
                    len_rubrics_average_partial.append(len_rubrics_average)
                elif c == 'INCORRECT':
                    relations_incorrect.append(rel)
                    len_rubrics_incorrect.append(len(len_rubrics))
                    len_rubrics_average_incorrect.append(len_rubrics_average)

        print('Average Values')
        print(language.upper() + ' Relation:', 'CORRECT', np.average(relations_correct), 'PARTIAL_CORRECT',
              np.average(relations_partial), 'INCORRECT', np.average(relations_incorrect))
        #plot_bars(['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT'], [np.average(relations_correct), np.average(relations_partial), np.average(relations_incorrect)], mode.upper() + '-' + language.upper() + ' Relation')
        print(language.upper() + ' Average len (tokens):', 'CORRECT', np.average(len_rubrics_average_correct),
              'PARTIAL_CORRECT', np.average(len_rubrics_average_partial), 'INCORRECT',
              np.average(len_rubrics_average_incorrect))
        #plot_bars(['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT'], [np.average(len_rubrics_average_correct), np.average(len_rubrics_average_partial), np.average(len_rubrics_average_incorrect)], mode.upper() + '-' + language.upper() + ' Average len (tokens)')
        print(language.upper() + ' Average number of rubrics in answer:', 'CORRECT', np.average(len_rubrics_correct),
              'PARTIAL_CORRECT', np.average(len_rubrics_partial), 'INCORRECT', np.average(len_rubrics_incorrect))
        plot_bars(['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT'], [np.average(len_rubrics_correct), np.average(len_rubrics_partial), np.average(len_rubrics_incorrect)], mode.upper() + '-' + language.upper() + ' Average number of rubrics in answer')
        print('Median Values')
        # print(language.upper() +' Relation:','CORRECT', np.median(relations_correct), 'PARTIAL_CORRECT', np.median(relations_partial),'INCORRECT', np.median(relations_incorrect))
        # print(language.upper() + ' Average len (tokens):', 'CORRECT',np.median(len_rubrics_average_correct),'PARTIAL_CORRECT', np.median(len_rubrics_average_partial), 'INCORRECT', np.median(len_rubrics_average_incorrect))
        print(language.upper() + ' Average number of rubrics in answer:', 'CORRECT', np.median(len_rubrics_correct),
              'PARTIAL_CORRECT', np.median(len_rubrics_partial), 'INCORRECT', np.median(len_rubrics_incorrect))
        #plot_bars(['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT'], [np.median(len_rubrics_correct), np.median(len_rubrics_partial), np.median(len_rubrics_incorrect)], mode.upper() + '-' + language.upper() + ' Median number of rubrics in answer')
        #plot_hist({'CORRECT': len_rubrics_average_correct, 'PARTIAL_CORRECT': len_rubrics_average_partial,
        #           'INCORRECT': len_rubrics_average_incorrect}, bins=range(0, 50, 5),
        #          title=language.upper() + ' Number of tokens in key element')
        #plot_hist(
        #    {'CORRECT': len_rubrics_correct, 'PARTIAL_CORRECT': len_rubrics_partial, 'INCORRECT': len_rubrics_incorrect},
        #    bins=range(0, 20, 2), title=language.upper() + ' Number of key elements in answer')
        results.append({
            'id': mode.upper() + '-CORRECT-' + language.upper(),
            'avg_relation': np.average(relations_correct),
            'avg_number_of_tokens_per_element': np.average(len_rubrics_average_correct),
            'avg_number_of_key_elements': np.average(len_rubrics_correct),
            'median_of_key_elements': np.median(len_rubrics_correct)
        })
        results.append({
            'id': mode.upper() + '-PARTIAL_CORRECT-' + language.upper(),
            'avg_relation': np.average(relations_partial),
            'avg_number_of_tokens_per_element': np.average(len_rubrics_average_partial),
            'avg_number_of_key_elements': np.average(len_rubrics_partial),
            'median_of_key_elements': np.median(len_rubrics_partial)
        })
        results.append({
            'id': mode.upper() + '-INCORRECT-' + language.upper(),
            'avg_relation': np.average(relations_incorrect),
            'avg_number_of_tokens_per_element': np.average(len_rubrics_average_incorrect),
            'avg_number_of_key_elements': np.average(len_rubrics_incorrect),
            'median_of_key_elements': np.median(len_rubrics_incorrect)
        })
    for lang in ['de', 'en']:
            stats = {
                'CORRECT': [],
                'PARTIAL_CORRECT': [],
                'INCORRECT': []
            }
            for an, label in zip(annotated_train_data, silver_labels):
                c = an['label']
                l = an['lang']
                if l == lang:
                    # for mode in ['average', 'max', 'average_nonzero', 'sum']:
                    stats[c].append(label)

            bin_items = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # counts, bins = np.histogram(x,bins=bin_items)
            plt.title(lang.upper() + ' ' + mode)
            plt.xlabel('Data')
            plt.ylabel('Frequency')
            correct = utils.flat_list(stats['CORRECT'])
            partial_correct = utils.flat_list(stats['PARTIAL_CORRECT'])
            incorrect = utils.flat_list(stats['INCORRECT'])
            normalized_data = []
            weights = []

            for d in [correct, partial_correct, incorrect]:
                w = np.ones_like(d) / len(d)
                weights.append(w)
            plt.hist([correct, partial_correct, incorrect], bins=bin_items, weights=weights,
                     label=['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT'])
            plt.xticks(bin_items)
            plt.legend()
            plt.show()
    print(10*'--')

    i = 0
    sample_size = 20
    color_map = {'0.1': '#FEFEF6',
                 '0.2': '#FBFBDC',
                 '0.3': '#FCFCAC',
                 '0.4': '#FBFBDC',
                 '0.5': '#FDFD84',
                 '0.6': '#F7F766',
                 '0.7': '#ECEC44',
                 '0.8': '#E8E834',
                 '0.9': '#EBEB1D',
                 '1.0': '#F5F503',
                 }


    def create_averaged_labels(indicies, raw_labels):
        labels = raw_labels.copy()
        for idx in indicies:
            s, e = idx[0], idx[1]
            label_value = round(np.average(labels[s:e]), 1)
            for i in range(s, e):
                labels[i] = label_value
        return labels


    def create_ents(tokens, indicies, labels=None):
        ents = []
        for idx in indicies:
            text = tokens[idx[0]:idx[1]]
            # s,e = text.start, text.end
            s, e = idx[0], idx[1]
            if labels != None:
                ents.append((str(round(np.average(labels[s:e]), 1)), s, e))
            else:
                ents.append(('CUE', s, e))
        return ents


    def get_idxs_elements(labels):
        indicies = []
        prev = 0
        start_i, end_i = 0, 0
        for i, l in enumerate(labels):
            if l == 0 and prev == 1:
                end_i = i
                indicies.append((start_i, end_i))
            if l == 1 and prev == 0:
                end_i = 0
                start_i = i
            if i == len(labels) - 1:  # end of sequence
                if l == 1:
                    end_i = i + 1  # because we want to get the last element as well
                    indicies.append((start_i, end_i))
            prev = l
        return indicies


    doc = []
    for a, raw_labels, in zip(annotated_train_data, silver_labels):
        text = a['student_answer']
        qid = a['question_id']
        question = a['question']
        l = a['label']
        hard_labels = silver2target(raw_labels, th=th)
        indicies = get_idxs_elements(hard_labels)
        # print(labels)
        if qid in german_question_ids:
            tokens = config.nlp_de(text)
        else:
            tokens = config.nlp(text)
        labels = create_averaged_labels(indicies, raw_labels)
        ents = create_ents(tokens, indicies, labels)
        tokens.ents = ents
        doc.append(tokens)

    skweak.utils.docbin_writer(doc, 'results/' + results_name + '_' + mode + '.spacy')
    print('saved to disk')

results = pd.DataFrame(columns=results[0].keys(), data=results)
results.to_csv('results/' + results_name + '.csv', index=False)

