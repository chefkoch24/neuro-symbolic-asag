import skweak

from aggregation import *
from config import Config

file_names_hmm = ['dev_ws_hmm_0', 'dev_ws_hmm_1', 'dev_ws_hmm_2']

config = Config()
for aggregation_method in ['sum', 'max', 'average', 'average_nonzero', 'hmm']:
    if aggregation_method != 'hmm':
        for file_name in ['training_ws_lfs', 'dev_ws_lfs']:
            data = utils.load_json(config.PATH_DATA + '/' + file_name + '.json')
            aggregation = Aggregation(config)
            aggregation.aggregate_labels(data, aggregation_method, 'aggregated_' + file_name + '_' + aggregation_method)
    # For HMM
    else:
        for file_name in file_names_hmm:
            data = utils.load_json(config.PATH_DATA + '/' + file_name + '.json')
            aggregation = AggregationHMM(config)
            docs = skweak.utils.docbin_reader('corpora/' + file_name + '.spacy', spacy_model_name='en_core_web_lg')
            aggregation.aggregate_labels(docs, data, 'aggregated_' + file_name)


