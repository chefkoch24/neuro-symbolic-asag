# Aggregation of the labeling functions to create the silver labels
import numpy as np

import config
import myutils as utils
# TODO: refine after analysis of softlabels
# Functions
def mean_nonzero(col):
    return np.mean([c for c in col if c!=0])

def aggregate_soft_labels(soft_labels, mode:str):
    if mode == 'average':
        return np.average(soft_labels, axis=0)
    elif mode == 'max':
        return np.max(soft_labels, axis=0)
    elif mode == 'average_nonzero':
        return np.apply_along_axis(mean_nonzero, axis=0, arr=soft_labels)
    elif mode == 'sum':
        return np.sum(soft_labels, axis=0)

def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(round(temp,1))
    return norm_arr

def extract_annotations(annotated_data, exclude_LFs=[]):
    labels = []
    for a in annotated_data:
        soft_labels = []
        for k,v in a['labeling_functions'].items():
            if k not in exclude_LFs:
                soft_labels.append(v)
        labels.append(soft_labels)
    return labels

def flat_list(lst):
    x = [item for sublist in lst for item in sublist]
    return x

def silver2target(data, th=0.5):
    targets = []
    for l in data:
        if l >= th:
            targets.append(1)
        else:
            targets.append(0)
    return targets


# Load data
annotated_train_data = utils.load_json(config.PATH_DATA + '/' +'train-soft.json')
annotated_dev_data = utils.load_json(config.PATH_DATA + '/' + 'dev-soft.json')

# Aggregate annotations
exclude = []
annotations_train = extract_annotations(annotated_train_data, exclude_LFs=exclude)
annotations_dev = extract_annotations(annotated_dev_data, exclude_LFs=exclude)
for i,data in enumerate([annotations_train, annotations_dev]):
    silver_labels = []
    for a in data:
        y = aggregate_soft_labels(a, 'sum')
        y = normalize(y)
        silver_labels.append(y)
    if i == 0:
        silver_label_train = silver_labels
    elif i == 1:
        silver_label_dev = silver_labels

for a, labels in zip(annotated_train_data, silver_label_train):
    a['silver_labels'] = labels
for a, labels in zip(annotated_dev_data, silver_label_dev):
    a['silver_labels'] = labels

# Save data
utils.save_json(annotated_train_data, config.PATH_DATA, 'train_labeled_data.json')
utils.save_json(annotated_dev_data, config.PATH_DATA,  'dev_labeled_data.json')