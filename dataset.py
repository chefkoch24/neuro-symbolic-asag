# datasets of the project
#Imports
import torch
from torch.utils.data import Dataset


class CustomJustificationCueDataset(Dataset):
    def __init__(self, answers, silver_labels, annotated_data):
        self.answers = []
        self.max_len = 512
        # 512 is hard coded as max length
        for i,a in enumerate(answers):
            pad_len = self.max_len - len(a['input_ids'])
            # add silverlabels as labels for the trainer
            a['input_ids'] = torch.nn.functional.pad(torch.tensor(a['input_ids']), pad=(0, pad_len), mode='constant', value=1) #padding token for distilroberta-base = 1
            a['attention_mask'] = torch.nn.functional.pad(torch.tensor(a['attention_mask']), pad=(0, pad_len), mode='constant', value=0)
            a['labels'] = torch.nn.functional.pad(torch.tensor(silver_labels[i]), pad=(0, pad_len), mode='constant', value=-100)
            a['label'] = annotated_data[i]['label']
            a['question_id'] = annotated_data[i]['question_id']
            self.answers.append(a)

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return self.answers[idx]