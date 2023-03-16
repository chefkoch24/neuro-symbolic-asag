#Imports
import time
import metrics
import numpy as np
import pandas as pd
from config import Config
import skweak
import myutils as utils
import logging

def evaluate_weak_supervision_models(files, config, th=0.5):
    results = []
    german_question_ids = [str(i) for i in range(1, 10)]
    for file in files:
        logging.info('Analyzing.. ' + file)
        disable_lang_filter = False
        annotated_data = utils.load_json(config.PATH_DATA + '/' + file)
        #Creating results
        for language in ['de', 'en', 'combined']:
            true_labels = []
            real_labels = []
            classes = []
            if language == 'combined':
                disable_lang_filter = True
            for an in annotated_data:
                lang = an['lang']
                c = an['label']
                labels = an['silver_labels']
                if lang == language or disable_lang_filter:
                    real_labels.append(labels)
                    true_label = metrics.silver2target(labels, th=th)
                    true_labels.append(true_label)
                    classes.append(c)
            # statistical metrics
            logging.info('Statistical Metrics')
            statistical_metrics= metrics.get_statistical_metrics(true_labels, classes)
            # Custom metrics
            n_correct, n_partial, n_incorrect = metrics.get_average_number_of_key_elements_by_class(true_labels, classes)
            r_correct, r_partial, r_incorrect = metrics.get_average_realtion_by_class(true_labels, classes)
            tn_correct, tn_partial, tn_incorrect = metrics.get_average_number_of_tokens_per_key_element_by_class(true_labels, classes)
            logging.info('Custom Metrics')
            logging.info(language.upper() + ' Relation:', 'CORRECT', r_correct, 'PARTIAL_CORRECT',
                  r_partial, 'INCORRECT', r_incorrect)
            logging.info(language.upper() + ' Average len (tokens):', 'CORRECT', tn_correct, 'PARTIAL_CORRECT', tn_partial, 'INCORRECT', tn_incorrect)
            logging.info(language.upper() + ' Average number of rubrics in answer:', 'CORRECT', n_correct,
                  'PARTIAL_CORRECT', n_partial, 'INCORRECT', n_incorrect)
            results.append({
                'id': file + '-CORRECT-' + language.upper(),
                'avg_relation': r_correct,
                'avg_number_of_tokens_per_element': tn_correct,
                'avg_number_of_key_elements': n_correct,
                'avg': statistical_metrics['average_correct'],
                'std': statistical_metrics['std_correct'],
                'median': statistical_metrics['median_correct'],
                'mode': statistical_metrics['mode_correct'],
                'min': statistical_metrics['min_correct'],
                'max': statistical_metrics['max_correct'],
                'labeled_tokens': statistical_metrics['labeled_tokens_correct'],
            })
            results.append({
                'id': file + '-PARTIAL_CORRECT-' + language.upper(),
                'avg_relation': r_partial,
                'avg_number_of_tokens_per_element': tn_partial,
                'avg_number_of_key_elements': n_partial,
                'avg': statistical_metrics['average_partial'],
                'std': statistical_metrics['std_partial'],
                'median': statistical_metrics['median_partial'],
                'mode': statistical_metrics['mode_partial'],
                'min': statistical_metrics['min_partial'],
                'max': statistical_metrics['max_partial'],
                'labeled_tokens': statistical_metrics['labeled_tokens_partial'],
            })
            results.append({
                'id': file + '-INCORRECT-' + language.upper(),
                'avg_relation': r_incorrect,
                'avg_number_of_tokens_per_element': tn_incorrect,
                'avg_number_of_key_elements': n_incorrect,
                'avg': statistical_metrics['average_incorrect'],
                'std': statistical_metrics['std_incorrect'],
                'median': statistical_metrics['median_incorrect'],
                'mode': statistical_metrics['mode_incorrect'],
                'min': statistical_metrics['min_incorrect'],
                'max': statistical_metrics['max_incorrect'],
                'labeled_tokens': statistical_metrics['labeled_tokens_incorrect'],
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
        for a in annotated_data:
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

    return results, doc


config = Config()
files = ['aggregated_dev_ws_lfs_average.json', 'aggregated_dev_ws_lfs_average_nonzero.json',
         'aggregated_dev_ws_lfs_max.json', 'aggregated_dev_ws_lfs_sum.json',
         'aggregated_dev_ws_hmm.json'
         ]
# investigate different thresholds
files = ['aggregated_dev_ws_hmm_0.json', 'aggregated_dev_ws_hmm_1.json', 'aggregated_dev_ws_hmm_2.json']

results, doc = evaluate_weak_supervision_models(files, config)
utils.save_csv(results, config.PATH_RESULTS, 'ws_results')