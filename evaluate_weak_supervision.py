# Settings
exclude = []
# exclude all hard labeling functions
#exclude = ['LF_lemma_match_without_stopwords', 'LF_pos_match_without_stopwords', 'LF_dep_match', 'LF_pos_match', 'LF_tag_match', 'LF_bleu_candidates','LF_edit_distance', 'LF_jaccard_similarity']
# exclude functions with low performance
#exclude = ['LF_pos_match', 'LF_tag_match', 'LF_dep_match','LF_pos_match_without_stopwords','LF_edit_distance', 'LF_jaccard_similarity']
th = 0.5
GLOBAL_NORMALIZE = False
#average_outliers = False

#Imports
import metrics
import numpy as np
import pandas as pd
import config
import skweak
import myutils as utils

def main():
    results = []
    german_question_ids = [str(i) for i in range(1, 10)]
    files = ['train_labeled_data_hmm.json', 'train_labeled_data_sum.json', 'train_labeled_data_average.json', 'train_labeled_data_max.json', 'train_labeled_data_average_nonzero.json']
    if GLOBAL_NORMALIZE:
        files = ['train_labeled_data_hmm.json', 'train_labeled_data_sum_global.json', 'train_labeled_data_average_global.json', 'train_labeled_data_max_global.json', 'train_labeled_data_average_nonzero_global.json']
    for file in files:
        print('Analyzing.. ' + file)
        disable_lang_filter = False
        annotated_train_data = utils.load_json(config.PATH_DATA + '/' + file)
        #Creating results
        for language in ['de', 'en', 'combined']:
            true_labels = []
            real_labels = []
            classes = []
            if language == 'combined':
                disable_lang_filter = True
            for an in annotated_train_data:
                lang = an['lang']
                c = an['label']
                labels = an['silver_labels']
                if lang == language or disable_lang_filter:
                    real_labels.append(labels)
                    true_label = metrics.silver2target(labels, th=th)
                    true_labels.append(true_label)
                    classes.append(c)
            # statistical metrics
            print('Statistical Metrics')
            statistical_metrics= metrics.get_statistical_metrics(true_labels, classes)
            # Custom metrics
            n_correct, n_partial, n_incorrect = metrics.get_average_number_of_key_elements_by_class(true_labels, classes)
            r_correct, r_partial, r_incorrect = metrics.get_average_realtion_by_class(true_labels, classes)
            tn_correct, tn_partial, tn_incorrect = metrics.get_average_number_of_tokens_per_key_element_by_class(true_labels, classes)
            print('Custom Metrics')
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
                'avg': statistical_metrics['average_correct'],
                'std': statistical_metrics['std_correct'],
                'median': statistical_metrics['median_correct'],
                'mode': statistical_metrics['mode_correct'],
                'min': statistical_metrics['min_correct'],
                'max': statistical_metrics['max_correct'],
                'support': statistical_metrics['support_correct'],
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
                'support': statistical_metrics['support_partial'],
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
                'support': statistical_metrics['support_incorrect'],
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
        if GLOBAL_NORMALIZE:
            results_name += '_global'
        skweak.utils.docbin_writer(doc, 'corpora/' + results_name + '.spacy')
        print('saved to disk')

    results = pd.DataFrame(columns=results[0].keys(), data=results)
    result_file_name = 'ws_results'
    if GLOBAL_NORMALIZE:
        result_file_name += '_global'
    results.to_csv('results/' + result_file_name + '.csv', index=False)

if __name__ == '__main__':
    main()