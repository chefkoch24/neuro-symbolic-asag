import argparse
import numpy as np
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from model import TokenClassificationModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import logging
import torch
import config
import myutils as utils
import warnings
from dataset import IterativeJustificationCueDataset
from transformers import AutoTokenizer
import metrics

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")



#Set seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

def get_spans(labels):
    true_labels = [l[1] for l in labels]
    hard_labels = metrics.silver2target(true_labels)
    spans = metrics.get_spans_from_labels(hard_labels)
    return spans

def get_rubric_elements(spans, input_ids, qid):
    decoded = [tokenizer.decode(i, skip_special_tokens=True) for i in input_ids]
    rubric = rubrics[qid]
    rubric_elements = []
    for s in spans:
        span_text = decoded[s[0]:s[1]]
        sim = para_detector.detect_paraphrases(span_text, rubric)
        max_index = np.argmax(sim)
        rubric_element = rubric['key_element'][max_index]
        rubric_elements.append(rubric_element)
    return rubric_elements

def generate_iterative_dataset(data):
    dataset = []
    for i, inputs in tqdm(enumerate(data), total=len(data)):
        input_ids = inputs['input_ids']
        student_answer = tokenizer.decode(input_ids, skip_special_tokens=True)
        spans = get_spans(inputs['labels'])
        q_id = inputs['question_id']
        rubric_elements = get_rubric_elements(spans, input_ids, q_id)
        # create a new entry for each span
        for span, re in zip(spans,rubric_elements):
            input = tokenizer(student_answer, add_special_tokens=False)
            input_len = len(input['input_ids'])
            label = np.zeros((input_len,))
            for i, l in enumerate(label):
                if i >= span[0] and i < span[1]:
                    label[i] = inputs['labels'][i][1]
            input = tokenizer(student_answer, re, add_special_tokens=True, padding='max_length', truncation=True, max_length=config.MAX_LEN)
            final_input_len = len(input['input_ids'])
            labels = label.tolist() + ([-100] * (final_input_len - input_len))
            labels[0] = -100 #because the first item is the CLS token
            labels = utils.create_labels_probability_distribution(labels)
            item = {
                    'input_ids': input['input_ids'],
                    'attention_mask': input['attention_mask'],
                    'labels': labels,
                    'question_id': q_id,
                    'rubric_element': re,
                    'class': inputs['class'],
                }
            dataset.append(item)
    return dataset

        # inputs['rubric_elements'] = self._get_rubric_elements(spans, inputs['input_ids'], inputs['question_id'])

parser=argparse.ArgumentParser()

parser.add_argument("--model", help="Name of the pretrained model")
parser.add_argument("--train_file", help="train file")
parser.add_argument("--dev_file", help="dev file")
parser.add_argument("--test_file", help="test file")
args=parser.parse_args()

# Load data
training_data = utils.load_json(config.PATH_DATA + '/' + args.train_file)
dev_data = utils.load_json(config.PATH_DATA + '/' + args.dev_file)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

# Preprocess data
para_detector = utils.ParaphraseDetector()
tokenizer = AutoTokenizer.from_pretrained(args.model)

preprocessed_training_data = generate_iterative_dataset(training_data)
preprocessed_dev_data = generate_iterative_dataset(dev_data)

#save data
DATASET_NAME = 'dataset_iterative_' + args.model+ '.json'
utils.save_json(preprocessed_training_data, config.PATH_DATA + '/', 'training_' + DATASET_NAME)
utils.save_json(preprocessed_dev_data, config.PATH_DATA + '/', 'dev_'+DATASET_NAME)