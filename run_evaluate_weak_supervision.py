#Imports
import os
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
    for file in files:
        logging.info('Analyzing.. ' + file)
        disable_lang_filter = False
        annotated_data = utils.load_json(config.PATH_DATA + '/aggregated/dev/' + file)
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
            file = file.split('.')[0]
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

    return results


config = Config()
files = [f for f in os.listdir('data/aggregated/dev')]
results = evaluate_weak_supervision_models(files, config)
utils.save_csv(results, config.PATH_RESULTS, 'ws_results')