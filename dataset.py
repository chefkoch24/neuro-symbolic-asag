# datasets of the project
#Imports
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


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rubrics = np.unique([d['question_id'] for d in self.dataset])
        self.filtered_data = []

    def __iter__(self):
        filtered_data = []
        for r in self.rubrics:
            data = [i for i, d in enumerate(self.dataset) if d['question_id'] == r]
            if len(data) > 0:
                filtered_data.append(data)
        data = []
        for fd in filtered_data:
            data += chunk(fd, self.batch_size)
        combined = [batch.tolist() for batch in data]
        self.filtered_data = filtered_data
        return iter(combined)
