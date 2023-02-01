# datasets of the project
#Imports
import torch
from torch.utils.data import Dataset


class CustomJustificationCueDataset(Dataset):
    def __init__(self, data, bert_tokens, silver_labels):
        self.model_inputs = []
        self.max_len = 512
        # 512 is hard coded as max length
        for i,inputs in enumerate(bert_tokens):
            pad_len = self.max_len - len(inputs['input_ids'])
            # add silverlabels as labels for the trainer
            inputs['input_ids'] = torch.nn.functional.pad(torch.tensor(inputs['input_ids']), pad=(0, pad_len), mode='constant', value=1) #padding token for distilroberta-base = 1
            inputs['attention_mask'] = torch.nn.functional.pad(torch.tensor(inputs['attention_mask']), pad=(0, pad_len), mode='constant', value=0)
            inputs['labels'] = torch.nn.functional.pad(torch.tensor(silver_labels[i]), pad=(0, pad_len), mode='constant', value=-100)
            inputs['class'] = data[i]['label']
            inputs['question_id'] = data[i]['question_id']
            self.model_inputs.append(inputs)

    def __len__(self):
        return len(self.model_inputs)

    def __getitem__(self, idx):
        return self.model_inputs[idx]