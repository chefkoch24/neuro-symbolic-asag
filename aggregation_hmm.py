import skweak
from spacy import displacy
import os

import config
import myutils as utils
import numpy as np

def append_silver_label(docs, data):
    for doc, d in zip(docs, data):
        label = np.zeros((len(doc)))
        probs = doc.spans["hmm"].attrs['probs']
        for k, v in probs.items():
            label[int(k)] = v['I-CUE']
        d['silver_labels'] = label.tolist()
    return data


annotated_train_data = utils.load_json(config.PATH_DATA + '/' + 'training_ws_hmm.json')
annotated_dev_data = utils.load_json(config.PATH_DATA + '/' + 'dev_ws_hmm.json')
train_corpus = skweak.utils.docbin_reader('corpora/' + 'corpora/train_labeled_data_hmm.spacy', spacy_model_name='en_core_web_lg')
dev_corpus = skweak.utils.docbin_reader('corpora/' + 'dev_labeled_data_hmm.spacy', spacy_model_name='en_core_web_lg')


train_corpus = list(train_corpus)
dev_corpus = list(dev_corpus)

annotated_train_data = append_silver_label(train_corpus, annotated_train_data)
annotated_dev_data = append_silver_label(dev_corpus, annotated_dev_data)

# Save data
utils.save_json(annotated_train_data, config.PATH_DATA, 'train_labeled_data_hmm.json')
utils.save_json(annotated_dev_data, config.PATH_DATA,  'dev_labeled_data_hmm.json')
