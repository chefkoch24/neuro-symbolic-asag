import numpy as np
import pandas as pd
from tqdm import tqdm

import myutils
from config import Config
from weak_supervision import WeakSupervisionSoft
from weak_supervision_hmm import WeakSupervisionHMM


def apply_lfs(data, model, rubrics, file_name, path):
    annotated_data = []
    for i, d in tqdm(data.iterrows()):
        q_id = d['question_id']
        x = d['tokenized']
        lang = d['lang']
        rubric = rubrics[q_id]
        len_seq = len(x)
        labeling_functions = {}
        for i, lf in enumerate(model.labeling_functions):
            soft_labels = np.zeros((len_seq))
            for cue in lf['function'](x, rubric, lang):
                soft_labels[cue[0]:cue[1] + 1] = cue[3]  # 3 is idx for soft label
            labeling_functions[lf['name']] = soft_labels.tolist()
        item = {
                'lang': d['lang'],
                'question_id': d['question_id'],
                'question': d['question'],
                'reference_answer': d['reference_answer'],
                'score': d['score'],
                'label': d['label'],
                'student_answer': d['student_answer'],
                'labeling_functions': labeling_functions,
        }
        annotated_data.append(item)
    myutils.save_json(annotated_data, path, file_name + '.json')

def apply_hmm(data, model, rubrics, file_name, path):
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
    myutils.save_json(annotated_data, path, file_name +  '.json')


# LOAD DATA
config = Config()
rubrics = myutils.load_json(config.PATH_RUBRIC)
rubrics = myutils.prepare_rubrics(rubrics)
X_train = pd.read_json(config.PATH_DATA + '/' + 'training_dataset.json')
X_dev = pd.read_json(config.PATH_DATA + '/' + 'dev_dataset.json')
X_train = myutils.tokenize_data(X_train)
X_dev = myutils.tokenize_data(X_dev)

# STANDARD WEAK SUPERVISION
ws = WeakSupervisionSoft(rubrics=rubrics)
apply_lfs(X_train, ws, rubrics, file_name='training_ws_lfs', path=config.PATH_DATA)
apply_lfs(X_dev, ws, rubrics, file_name='dev_ws_lfs', path=config.PATH_DATA)

# HMM WEAK SUPERVISION
ws = WeakSupervisionHMM(rubrics=rubrics, meteor_th=0.05, ngram_th=0.10, rouge_th=0.15, edit_dist_th=0.5,
                            paraphrase_th=0.9, bleu_th=0.5, jaccard_th=0.5, mode='hmm')
ws.fit(X_train)
apply_hmm(X_train, ws, rubrics, file_name='training_ws_hmm', path=config.PATH_DATA)
apply_hmm(X_dev, ws, rubrics, file_name='dev_ws_hmm', path=config.PATH_DATA)





