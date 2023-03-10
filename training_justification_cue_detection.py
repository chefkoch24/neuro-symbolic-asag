# Imports
import argparse
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from transformers import AutoTokenizer

from model import TokenClassificationModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import logging
import torch
import config
import myutils as utils
logging.basicConfig(level=logging.ERROR)
from dataset import JustificationCueDataset
import warnings
warnings.filterwarnings("ignore")

def pre_process(data, with_context=False):
    model_inputs = []
    for d in tqdm(data):
        student_answer = d['student_answer']
        # Tokenize the input
        if with_context:
            context = d['reference_answer']
            tokenized = tokenizer(student_answer, context, max_length=config.MAX_LEN, truncation=True, padding='max_length', return_token_type_ids=True)
        else:
            tokenized = tokenizer(student_answer, max_length=config.MAX_LEN, truncation=True, padding='max_length', return_token_type_ids=True)
        # Generating the labels
        aligned_labels = d['aligned_labels']
        pad_len = config.MAX_LEN - len(aligned_labels) -2
        labels = [-100]+aligned_labels + [-100]
        # Adding other model inputs
        model_input = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'token_type_ids': tokenized['token_type_ids'],
            'labels': utils.create_labels_probability_distribution(torch.nn.functional.pad(torch.tensor(labels), pad=(0, pad_len), mode='constant', value=-100).detach().numpy().tolist()),
            'class': d['label'],
            'question_id': d['question_id'],
            'student_answer': d['student_answer'],
            'reference_answer': d['reference_answer'],
        }
        model_inputs.append(model_input)

    return model_inputs

#Set seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
# Load data
train_data = utils.load_json(config.PATH_DATA + '/' + config.ALIGNED_TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.ALIGNED_DEV_FILE)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

# Preprocess data
training_dataset = pre_process(train_data, with_context=config.CONTEXT)
dev_dataset = pre_process(dev_data, with_context=config.CONTEXT)

training_dataset = JustificationCueDataset(training_dataset)
dev_dataset = JustificationCueDataset(dev_dataset)
# Training
train_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
model = TokenClassificationModel(config.MODEL_NAME)


EXPERIMENT_NAME = "justification_cue" + "_" + config.MODEL_NAME + "_context-" + str(config.CONTEXT) + "_bs-" + str(config.BATCH_SIZE)
logger = CSVLogger("logs", name=EXPERIMENT_NAME)
trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  #gradient_clip_val=0.5,
                  #accumulate_grad_batches=2,
                  #auto_scale_batch_size='power',
                  callbacks=[
                      config.checkpoint_callback,
                      config.early_stop_callback
                             ],
                  logger=logger)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)