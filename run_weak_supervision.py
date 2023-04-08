import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import myutils
from config import Config
from weak_supervision import WeakSupervisionSoft
from weak_supervision_hmm import WeakSupervisionHMM

def apply_hmm(data, model, file_name, path):
    result = model.predict(data)
    myutils.save_annotated_corpus(result, "corpora/" + file_name + ".spacy")
    annotated_data = []
    for i, d in tqdm(data.iterrows()):
        item = {
                'lang': d['lang'],
                'question_id': d['question_id'],
                'question': d['question'],
                'reference_answer': d['reference_answer'],
                'score': d['score'],
                'label': d['label'],
                'student_answer': d['student_answer'],
                'labeling_functions': {},
        }
        annotated_data.append(item)
    myutils.save_json(annotated_data, path, file_name + '.json')
    return annotated_data, result

# LOAD DATA
logging.info("Loading data...")
config = Config()
rubrics = myutils.load_rubrics(config.PATH_RUBRIC)
rubrics = myutils.prepare_rubrics(rubrics, config)
X_train = pd.read_json(config.PATH_DATA + '/' + 'training_dataset.json')
X_dev = pd.read_json(config.PATH_DATA + '/' + 'dev_dataset.json')
X_train = myutils.tokenize_data(X_train, config)
X_dev = myutils.tokenize_data(X_dev, config)

# HMM WEAK SUPERVISION
thresholds = {
    'meteor': 0.4,
    'ngram': 0.4,
    'rouge': 0.4,
    'edit_dist': 0.5,
    'paraphrase': 0.9,
    'bleu': 0.4,
    'jaccard': 0.5,
}
logging.info("Start HMM Weak Supervision...")
ws = WeakSupervisionHMM(rubrics=rubrics, meteor_th=thresholds['meteor'], ngram_th=thresholds['ngram'],
                            rouge_th=thresholds['rouge'], edit_dist_th=thresholds['edit_dist'],
                            paraphrase_th=thresholds['paraphrase'], bleu_th=thresholds['bleu'],
                            jaccard_th=thresholds['jaccard'], mode='hmm', config=config)
ws.fit(X_train)
apply_hmm(X_train, ws, file_name='training_ws_hmm', path=config.PATH_DATA)
apply_hmm(X_dev, ws, file_name='dev_ws_hmm', path=config.PATH_DATA)




