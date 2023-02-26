# Aggregation of the labeling functions to create the silver labels
import numpy as np
import config
import myutils as utils

GLOBAL_NORMALIZE = False

# Functions
def mean_nonzero(col):
    return np.mean([c for c in col if c!=0])

def aggregate_soft_labels(soft_labels, mode:str):
    if mode == 'average':
        return np.average(soft_labels, axis=0).tolist()
    elif mode == 'max':
        return np.max(soft_labels, axis=0).tolist()
    elif mode == 'average_nonzero':
        return np.apply_along_axis(mean_nonzero, axis=0, arr=soft_labels).tolist()
    elif mode == 'sum':
        return np.sum(soft_labels, axis=0).tolist()

def normalize(sequence):
    min_value = min(sequence)
    max_value = max(sequence)
    range_value = max_value - min_value
    if range_value == 0:
        range_value = 0.001
    normalized_sequence = [(x - min_value) / range_value for x in sequence]
    return normalized_sequence

def extract_annotations(annotated_data, exclude_LFs=[]):
    labels = []
    for a in annotated_data:
        soft_labels = []
        for k,v in a['labeling_functions'].items():
            if k not in exclude_LFs:
                soft_labels.append(v)
        labels.append(soft_labels)
    return labels

def global_normalize(data, average_outliers=False):
    # finding min and max value
    min_value = 100
    max_value = -100
    max_vals = []
    min_vals = []
    for d in data:
        v_min = np.min(d)
        v_max = np.max(d)
        if v_min < min_value:
            min_value = v_min
            min_vals.append(min_value)
        if v_max > max_value:
            max_value = v_max
            max_vals.append(max_value)
    if average_outliers:
        min_value = np.average(min_vals)
        max_value = np.average(max_vals)
    range_val = max_value - min_value
    data_norm = [[(l - min_value) / range_val for l in d] for d in data]
    return data_norm

# Load data
def _main():
    annotated_train_data = utils.load_json(config.PATH_DATA + '/' +'training_ws_lfs.json')
    annotated_dev_data = utils.load_json(config.PATH_DATA + '/' + 'dev_ws_lfs.json')

    # Aggregate annotations
    exclude = []
    for mode in ['average', 'max', 'average_nonzero', 'sum']:
        annotations_train = extract_annotations(annotated_train_data, exclude_LFs=exclude)
        annotations_dev = extract_annotations(annotated_dev_data, exclude_LFs=exclude)
        for i,data in enumerate([annotations_train, annotations_dev]):
            silver_labels = []
            for a in data:
                y = aggregate_soft_labels(a, mode)
                if not GLOBAL_NORMALIZE:
                    if mode == 'sum':
                        y = normalize(y)
                    else:
                        y = y
                silver_labels.append(y)
            if GLOBAL_NORMALIZE:
                silver_labels = global_normalize(silver_labels, average_outliers=False)
            if i == 0:
                silver_label_train = silver_labels
            elif i == 1:
                silver_label_dev = silver_labels

        for a, labels in zip(annotated_train_data, silver_label_train):
            a['silver_labels'] = labels
        for a, labels in zip(annotated_dev_data, silver_label_dev):
            a['silver_labels'] = labels

        # Save data
        if GLOBAL_NORMALIZE:
            mode = mode + '_global'
        utils.save_json(annotated_train_data, config.PATH_DATA, 'train_labeled_data_' + mode + '.json')
        utils.save_json(annotated_dev_data, config.PATH_DATA,  'dev_labeled_data_' + mode + '.json')

if __name__ == '__main__':
    _main()