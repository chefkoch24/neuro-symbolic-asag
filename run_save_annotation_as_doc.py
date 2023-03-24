import numpy as np
import skweak

import metrics
from config import Config
import myutils as utils
german_question_ids = [str(i) for i in range(1, 10)]

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
config = Config()
annotated_data = utils.load_json('data/aggregated/dev/dev_ws_lfs_sum_all_lfs.json')
for a in annotated_data:
    text = a['student_answer']
    qid = a['question_id']
    question = a['question']
    l = a['label']
    raw_labels = a['silver_labels']
    hard_labels = metrics.silver2target(raw_labels, th=0.5)
    spans = metrics.get_spans_from_labels(hard_labels)
    if qid in german_question_ids:
        tokens = config.nlp_de(text)
    else:
        tokens = config.nlp(text)
    labels = create_averaged_labels(spans, raw_labels)
    ents = create_ents(tokens, spans, labels)
    tokens.ents = ents
    doc.append(tokens)

skweak.utils.docbin_writer(doc, 'corpora/dev_ws_lfs_sum_all_lfs.spacy')