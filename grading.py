# This script finally grades the student answers
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForTokenClassification
from incremental_trees.models.classification.streaming_rfc import StreamingRFC

# Define the model checkpoint
import config
from dataset import GradingDataset
from grading_model import GradingModelClassification
import myutils as utils

model_checkpoint = 'logs/justification_cue_distilroberta-base_context-False/version_8/checkpoints/checkpoint-epoch=03-val_loss=0.45.ckpt'
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)


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
        for r in rubrics:
            data = [i for i, d in enumerate(self.dataset) if d['question_id'] == r]
            if len(data) > 0:
                filtered_data.append(data)
        data = []
        for fd in filtered_data:
            data += chunk(fd, self.batch_size)
        combined = [batch.tolist() for batch in data]
        self.filtered_data = filtered_data
        return iter(combined)

    def __len__(self):
        return sum([len(d) for d in self.filtered_data]) // self.batch_size

def preprocess(data, class2idx={'CORRECT': 0, 'PARTIAL_CORRECT': 1, 'INCORRECT': 2}):
    for d in data:
        tokenized = tokenizer(d['student_answer'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        d['input_ids'] = tokenized['input_ids']
        d['attention_mask'] = tokenized['attention_mask']
        d['class'] = class2idx[d['label']]
    return data

train_file = 'training_dataset.json'
dev_file = 'dev_dataset.json'
# Create dataset and dataloader
training_data = utils.load_json(config.PATH_DATA + '/' + train_file)
dev_data = utils.load_json(config.PATH_DATA + '/' + dev_file)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

training_data = preprocess(training_data)
dev_data = preprocess(dev_data)

training_dataset = GradingDataset(training_data)
dev_dataset = GradingDataset(dev_data)

train_loader = DataLoader(training_dataset, batch_sampler=CustomBatchSampler(training_dataset, config.BATCH_SIZE))
val_loader = DataLoader(dev_dataset, batch_sampler=CustomBatchSampler(dev_dataset, config.BATCH_SIZE))
model = GradingModelClassification(model_checkpoint, StreamingRFC(), rubrics=rubrics, model_name=config.MODEL_NAME)

EXPERIMENT_NAME = "grading"
logger = CSVLogger("logs", name=EXPERIMENT_NAME)
trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  #gradient_clip_val=0.5,
                  #accumulate_grad_batches=2,
                  #auto_scale_batch_size='power',
                  callbacks=[
                      config.checkpoint_callback,
                      # early_stop_callback
                             ],
                  logger=logger)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)