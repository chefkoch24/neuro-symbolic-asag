import skweak

from aggregation import *
from config import Config

config = Config()
for aggregation_method in ['sum', 'max', 'average', 'average_nonzero', 'hmm']:
    if aggregation_method != 'hmm':
        for file_name in ['training_ws_lfs', 'dev_ws_lfs']:
            data = utils.load_json(config.PATH_DATA + '/' + file_name + '.json')
            aggregation = Aggregation(config)
            aggregation.aggregate_labels(data, aggregation_method, 'aggregated_' + file_name + '_' + aggregation_method)
    # For HMM
    else:
        for file_name in ['training_ws_hmm', 'dev_ws_hmm']:
            data = utils.load_json(config.PATH_DATA + '/' + file_name + '.json')
            aggregation = AggregationHMM(config)
            docs = skweak.utils.docbin_reader('corpora/' + file_name + '.spacy', spacy_model_name=config.nlp)
            aggregation.aggregate_labels(data, docs,  'aggregated_' + file_name)

