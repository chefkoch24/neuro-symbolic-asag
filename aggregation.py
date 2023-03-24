# Aggregation of the labeling functions to create the silver labels
import numpy as np
import config
import myutils as utils


class Aggregation:

    def __init__(self, config):
        self.config = config

    # Functions
    def mean_nonzero(self, col):
        return np.mean([c for c in col if c!=0])

    def aggregate_soft_labels(self, soft_labels, mode:str):
        if mode == 'average':
            return np.average(soft_labels, axis=0).tolist()
        elif mode == 'max':
            return np.max(soft_labels, axis=0).tolist()
        elif mode == 'average_nonzero':
            return np.apply_along_axis(self.mean_nonzero, axis=0, arr=soft_labels).tolist()
        elif mode == 'sum':
            return np.sum(soft_labels, axis=0).tolist()

    def normalize(self, sequence):
        min_value = min(sequence)
        max_value = max(sequence)
        range_value = max_value - min_value
        if range_value == 0:
            range_value = 0.001
        normalized_sequence = [(x - min_value) / range_value for x in sequence]
        return normalized_sequence

    def extract_annotations(self, annotated_data, exclude_LFs=[]):
        labels = []
        for a in annotated_data:
            soft_labels = []
            for k,v in a['labeling_functions'].items():
                if k not in exclude_LFs:
                    soft_labels.append(v)
            labels.append(soft_labels)
        return labels

    def global_normalize(self, data, average_outliers=False):
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

    def aggregate_labels(self, data, mode:str):
        annotated_data = self.extract_annotations(data, exclude_LFs=self.config.EXCLUDED_LFS)
        silver_labels = []
        for a in annotated_data:
            y = self.aggregate_soft_labels(a, mode)
            if mode == 'sum':
                y = self.normalize(y)
            silver_labels.append(y)
        for a, labels in zip(data, silver_labels):
            a['silver_labels'] = labels
        return annotated_data

class AggregationHMM:

    def __init__(self, config):
        self.config = config

    def aggregate_labels(self, docs, data):
        annotated_data = []
        for doc, d in zip(docs, data):
            label = np.zeros((len(doc)))
            probs = doc.spans["hmm"].attrs['probs']
            for k, v in probs.items():
                label[int(k)] = v['I-CUE']
            d['silver_labels'] = label.tolist()
            annotated_data.append(d)
        return annotated_data
