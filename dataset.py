# datasets of the project
#Imports
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import config
import metrics
import myutils


class JustificationCueDataset(Dataset):
    def __init__(self, data):
        self.data = data
        for i,inputs in enumerate(self.data):
            inputs['input_ids'] = torch.tensor(inputs['input_ids'])
            inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
            inputs['labels'] = torch.tensor(inputs['labels'])

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GradingDataset(Dataset):
    def __init__(self, answer_texts, input_ids, question_ids, labels, scores):
        self.answer_texts = answer_texts
        self.question_ids = question_ids
        self.labels = labels
        self.scores = scores
        self.input_ids = input_ids

    def __len__(self):
        return len(self.answer_texts)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'answer_text':  self.answer_texts[idx],
            'question_id':  self.question_ids[idx],
            'label': self.labels[idx],
            'score': self.scores[idx]
            }
        return item