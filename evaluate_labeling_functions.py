# Settings
th = 0.5
GLOBAL_NORMALIZE = False

#Imports
import metrics
import pandas as pd
import config
import myutils as utils

def main():
    results = []
    files = ['training_ws_lfs.json']
    for file in files:
        print('Analyzing.. ' + file)
        annotated_train_data = utils.load_json(config.PATH_DATA + '/' + file)
        # setup statistic dicts
        # calculate stats splitted by German and English
        stats_de = {}
        stats_en = {}
        stats_combined = {}
        # initialize stats dict  for the individual labeling_functions
        for k in annotated_train_data[0]['labeling_functions']:
            stats_de[k] = {'class': [], 'labels': []}
        for k in annotated_train_data[0]['labeling_functions']:
            stats_en[k] = {'class': [], 'labels': []}
        for k in annotated_train_data[0]['labeling_functions']:
            stats_combined[k] = {'class': [], 'labels': []}

        for an in annotated_train_data:
            c = an['label']
            lang = an['lang']
            if lang == 'en':
                for k, v in an['labeling_functions'].items():
                    stats_en[k]['labels'].append(v)
                    stats_en[k]['class'].append(c)
            elif lang == 'de':
                for k, v in an['labeling_functions'].items():
                    stats_de[k]['labels'].append(v)
                    stats_de[k]['class'].append(c)
        for an in annotated_train_data:
            c = an['label']
            for k, v in an['labeling_functions'].items():
                stats_combined[k]['labels'].append(v)
                stats_combined[k]['class'].append(c)


        for language, stats in {'DE': stats_de, 'EN': stats_en, 'combined': stats_combined}.items():
            for k, v in stats.items():
                # statistical metrics
                statistical_metrics = metrics.get_statistical_metrics(v['labels'], v['class'])
                # Custom metrics
                true_labels = [metrics.silver2target(l, th=th) for l in v['labels']]
                n_correct, n_partial, n_incorrect = metrics.get_average_number_of_key_elements_by_class(true_labels, v['class'])
                r_correct, r_partial, r_incorrect = metrics.get_average_realtion_by_class(true_labels, v['class'])
                tn_correct, tn_partial, tn_incorrect = metrics.get_average_number_of_tokens_per_key_element_by_class(true_labels, v['class'])
                results.append({
                    'id': k + '-CORRECT-' + language.upper(),
                    'avg_relation': r_correct,
                    'avg_number_of_tokens_per_element': tn_correct,
                    'avg_number_of_key_elements': n_correct,
                    'avg': statistical_metrics['average_correct'],
                    'std': statistical_metrics['std_correct'],
                    'median': statistical_metrics['median_correct'],
                    'mode': statistical_metrics['mode_correct'],
                    'min': statistical_metrics['min_correct'],
                    'max': statistical_metrics['max_correct'],
                    'support': statistical_metrics['support_correct'],

                })
                results.append({
                    'id': k + '-PARTIAL_CORRECT-' + language.upper(),
                    'avg_relation': r_partial,
                    'avg_number_of_tokens_per_element': tn_partial,
                    'avg_number_of_key_elements': n_partial,
                    'avg': statistical_metrics['average_partial'],
                    'std': statistical_metrics['std_partial'],
                    'median': statistical_metrics['median_partial'],
                    'mode': statistical_metrics['mode_partial'],
                    'min': statistical_metrics['min_partial'],
                    'max': statistical_metrics['max_partial'],
                    'support': statistical_metrics['support_partial'],
                })
                results.append({
                    'id': k + '-INCORRECT-' + language.upper(),
                    'avg_relation': r_incorrect,
                    'avg_number_of_tokens_per_element': tn_incorrect,
                    'avg_number_of_key_elements': n_incorrect,
                    'avg': statistical_metrics['average_incorrect'],
                    'std': statistical_metrics['std_incorrect'],
                    'median': statistical_metrics['median_incorrect'],
                    'mode': statistical_metrics['mode_incorrect'],
                    'min': statistical_metrics['min_incorrect'],
                    'max': statistical_metrics['max_incorrect'],
                    'support': statistical_metrics['support_incorrect'],
                })

    results = pd.DataFrame(columns=results[0].keys(), data=results)
    result_file_name = 'labeling_functions_results'
    results.to_csv('results/' + result_file_name + '.csv', index=False)

if __name__ == '__main__':
    main()