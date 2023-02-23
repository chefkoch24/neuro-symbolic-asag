# Settings
import aggregation
import metrics

exclude = []
# exclude all hard labeling functions
#exclude = ['LF_lemma_match_without_stopwords', 'LF_pos_match_without_stopwords', 'LF_dep_match', 'LF_pos_match', 'LF_tag_match', 'LF_bleu_candidates','LF_edit_distance', 'LF_jaccard_similarity']
# exclude functions with low performance
#exclude = ['LF_pos_match', 'LF_tag_match', 'LF_dep_match','LF_pos_match_without_stopwords','LF_edit_distance', 'LF_jaccard_similarity']
TOKENIZER_NAME = 'distilroberta-base'
th = 0.5
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


def normalize(sequence):
    min_value = min(sequence)
    max_value = max(sequence)
    range_value = max_value - min_value
    if range_value == 0:
        range_value = 0.001
    normalized_sequence = [(x - min_value) / range_value for x in sequence]
    return normalized_sequence

def plot_bars(classes, data, title):
    plt.bar(classes, data)
    plt.title(title)
    plt.show()

# Loading
def main():
    results = []
    german_question_ids = [str(i) for i in range(1, 10)]
    files = ['train_labeled_data_hmm.json', 'train_labeled_data_sum.json', 'train_labeled_data_average.json', 'train_labeled_data_max.json', 'train_labeled_data_average_nonzero.json']
    rubrics = utils.load_rubrics(config.PATH_DATA + '/' + 'rubrics.json')
    for file in files:
        print('Analyzing.. ' + file)
        disable_lang_filter = False
        annotated_train_data = utils.load_json(config.PATH_DATA + '/' + file)
        #Creating results
        for language in ['de', 'en', 'combined']:
            true_labels = []
            classes = []
            if language == 'combined':
                disable_lang_filter = True
            for an in annotated_train_data:
                lang = an['lang']
                c = an['label']
                labels = an['silver_labels']
                if lang == language or disable_lang_filter:
                    true_label = metrics.silver2target(labels, th=th)
                    true_labels.append(true_label)
                    classes.append(c)
            n_correct, n_partial, n_incorrect = metrics.get_average_number_of_key_elements_by_class(true_labels, classes)
            r_correct, r_partial, r_incorrect = metrics.get_average_realtion_by_class(true_labels, classes)
            tn_correct, tn_partial, tn_incorrect = metrics.get_average_number_of_tokens_per_key_element_by_class(true_labels, classes)
            print('Average Values')
            print(language.upper() + ' Relation:', 'CORRECT', r_correct, 'PARTIAL_CORRECT',
                  r_partial, 'INCORRECT', r_incorrect)
            #plot_bars(['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT'], [np.average(relations_correct), np.average(relations_partial), np.average(relations_incorrect)], mode.upper() + '-' + language.upper() + ' Relation')
            print(language.upper() + ' Average len (tokens):', 'CORRECT', tn_correct, 'PARTIAL_CORRECT', tn_partial, 'INCORRECT', tn_incorrect)
            #plot_bars(['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT'], [np.average(len_rubrics_average_correct), np.average(len_rubrics_average_partial), np.average(len_rubrics_average_incorrect)], mode.upper() + '-' + language.upper() + ' Average len (tokens)')
            print(language.upper() + ' Average number of rubrics in answer:', 'CORRECT', n_correct,
                  'PARTIAL_CORRECT', n_partial, 'INCORRECT', n_incorrect)
            #plot_bars(['CORRECT', 'PARTIAL_CORRECT', 'INCORRECT'], [n_correct,n_partial, n_incorrect], language.upper() + ' Average number of rubrics in answer')
            results.append({
                'id': file + '-CORRECT-' + language.upper(),
                'avg_relation': r_correct,
                'avg_number_of_tokens_per_element': tn_correct,
                'avg_number_of_key_elements': n_correct,
            })
            results.append({
                'id': file + '-PARTIAL_CORRECT-' + language.upper(),
                'avg_relation': r_partial,
                'avg_number_of_tokens_per_element': tn_partial,
                'avg_number_of_key_elements': n_partial,
            })
            results.append({
                'id': file + '-INCORRECT-' + language.upper(),
                'avg_relation': r_incorrect,
                'avg_number_of_tokens_per_element': tn_incorrect,
                'avg_number_of_key_elements': n_incorrect,
            })

    # save as spacy doc
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
        doc = []
        for a in annotated_train_data:
            text = a['student_answer']
            qid = a['question_id']
            question = a['question']
            l = a['label']
            raw_labels = a['silver_labels']
            hard_labels = metrics.silver2target(raw_labels, th=th)
            spans = metrics.get_spans_from_labels(hard_labels)
            if qid in german_question_ids:
                tokens = config.nlp_de(text)
            else:
                tokens = config.nlp(text)
            labels = create_averaged_labels(spans, raw_labels)
            ents = create_ents(tokens, spans, labels)
            tokens.ents = ents
            doc.append(tokens)

        results_name = file.split('.')[0]
        skweak.utils.docbin_writer(doc, 'corpora/' + results_name + '.spacy')
        print('saved to disk')

    results = pd.DataFrame(columns=results[0].keys(), data=results)
    results.to_csv('results/results.csv', index=False)

if __name__ == '__main__':
    main()