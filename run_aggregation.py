import skweak

from aggregation import *
from config import Config

experiments = {
    'all_lfs': [],
    'non_hard_matches': [
    'LF_noun_phrases', 'LF_lemma_match', 'LF_pos_match', 'LF_shape_match', 'LF_stem_match',
    'LF_dep_match', 'LF_lemma_match_without_stopwords', 'LF_stem_match_without_stopwords',
    'LF_pos_match_without_stopwords', 'LF_dep_match_without_stopwords', 'LF_word_alignment'
    ]
}
for experiment_name, value in experiments.items():
    config = Config(
        excluded_lfs=value
    )
    for aggregation_method in ['hmm']:#['sum', 'max', 'average', 'average_nonzero', 'hmm']:
        for split in ['training', 'dev']:
            if aggregation_method != 'hmm':
                for file_name in ['ws_lfs', 'ws_lfs']:
                    data = utils.load_json(config.PATH_DATA + '/' + split + '_' + file_name + '.json')
                    aggregation = Aggregation(config)
                    annotated_data = aggregation.aggregate_labels(data, aggregation_method)
                    file_name = utils.get_experiment_name([split, file_name, aggregation_method, experiment_name])
                    utils.save_json(data, config.PATH_DATA + '/aggregated/' + split , file_name + '.json')

            # For HMM
            else:
                file_names_hmm = ['ws_hmm']#, 'ws_hmm_1', 'ws_hmm_2']
                for file_name in file_names_hmm:
                    file_name = split + '_' + file_name
                    data = utils.load_json(config.PATH_DATA + '/' + file_name + '.json')
                    aggregation = AggregationHMM(config)
                    docs = skweak.utils.docbin_reader('corpora/' + file_name + '.spacy', spacy_model_name='en_core_web_lg')
                    file_name = utils.get_experiment_name([file_name, experiment_name])
                    annotated_data = aggregation.aggregate_labels(docs, data)
                    utils.save_json(annotated_data, config.PATH_DATA + '/aggregated/' + split, file_name + '.json')

