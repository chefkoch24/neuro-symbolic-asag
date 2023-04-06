# datasets of the project
#Imports
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer

import config
import metrics
import myutils as utils


class JustificationCueDataset(Dataset):
    def __init__(self, data):
        self.data = data
        for i,inputs in enumerate(self.data):
            inputs['input_ids'] = torch.tensor(inputs['input_ids'])
            inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
            inputs['labels'] = torch.tensor(inputs['labels'])
            inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IterativeJustificationCueDataset(Dataset):
    def __init__(self, data):
        self.data = data
        for i, inputs in enumerate(self.data):
            inputs['input_ids'] = torch.tensor(inputs['input_ids'])
            inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
            inputs['labels'] = torch.tensor(inputs['labels'])
            inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SpanJustificationCueDataset(Dataset):
    def __init__(self, data):
        self.data = data
        for i, inputs in enumerate(self.data):
            inputs['input_ids'] = torch.tensor(inputs['input_ids'])
            inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
            inputs['start_positions'] = torch.tensor(inputs['start_positions'])
            inputs['end_positions'] = torch.tensor(inputs['end_positions'])
            inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GradingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size,seed=42):
        np.random.seed(seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.rubrics = np.unique([d['question_id'] for d in self.dataset])
        self.filtered_data = []
        for r in self.rubrics:
            data = [i for i, d in enumerate(self.dataset) if d['question_id'] == r]
            if len(data) > 0:
                self.filtered_data.append(data)

    def __iter__(self):
        combined = []
        for fd in self.filtered_data:
            batches = [fd[i:i+self.batch_size] for i in range(0, len(fd), self.batch_size)]
            combined += batches
        shuffled = np.random.permutation(combined)
        return iter(shuffled)

    def __len__(self):
        num_batches = sum([math.ceil(len(fd) / self.batch_size) for fd in self.filtered_data])
        return num_batches
