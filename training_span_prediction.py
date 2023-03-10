import numpy as np
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import metrics
import myutils as utils
import config
from dataset import SpanJustificationCueDataset
from model import SpanPredictionModel
from paraphrase_scorer import BertScorer


def get_rubric_elements(spans, input_ids, qid):
    rubric = rubrics[qid]
    rubric_elements = []
    sims = []
    for s in spans:
        span_text = tokenizer.decode(input_ids[s[0]:s[1]])
        sim = para_detector.detect_score_key_elements(span_text, rubric)
        max_index = np.argmax(sim)
        rubric_element = rubric['key_element'][max_index]
        rubric_elements.append(rubric_element)
        sims.append(np.max(sim))
    return rubric_elements, sims


def create_inputs(data):
    model_inputs = []
    for d in tqdm(data):
        student_answer = d['student_answer']
        aligned_labels = d['aligned_labels']
        q_id = d['question_id']
        # Tokenize the input to generate alignment
        tokenized = tokenizer(student_answer, add_special_tokens=False, return_offsets_mapping=True)
        # get the spans from the aligned labels
        hard_labels = metrics.silver2target(aligned_labels)
        spans = metrics.get_spans_from_labels(hard_labels)
        rubric_elements, rubric_sims = get_rubric_elements(spans, tokenized['input_ids'], q_id)
        # generate final input for every span individually
        for span, re, sim in zip(spans, rubric_elements, rubric_sims):
            tokenized = tokenizer(student_answer, re, max_length=config.MAX_LEN, truncation=True, padding='max_length', return_token_type_ids=True)
            model_input = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'token_type_ids': tokenized['token_type_ids'],
                'start_positions': [span[0]],
                'end_positions': [span[1]],
                'question_id': q_id,
                'rubric_element': re,
                'class': d['label'],
                'similarity': sim,
                'student_answer': student_answer
            }
            model_inputs.append(model_input)
    return model_inputs


#Set seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
#Loading
train_data = utils.load_json(config.PATH_DATA + '/' + config.ALIGNED_TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.ALIGNED_DEV_FILE)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
para_detector = BertScorer()


# Load data
training_data = utils.load_json(config.PATH_DATA + '/' + config.ALIGNED_TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.ALIGNED_DEV_FILE)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

training_dataset = SpanJustificationCueDataset(training_data)
dev_dataset = SpanJustificationCueDataset(dev_data)
# Training
train_loader = DataLoader(training_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
model = SpanPredictionModel(config.MODEL_NAME)

EXPERIMENT_NAME = "span_prediction" + "_" + config.MODEL_NAME + "_batch_" + str(config.BATCH_SIZE)
logger = CSVLogger("logs", name=EXPERIMENT_NAME)
trainer = Trainer(max_epochs=config.NUM_EPOCHS,
                  #gradient_clip_val=0.5,
                  #accumulate_grad_batches=2,
                  #auto_scale_batch_size='power',
                  callbacks=[config.checkpoint_callback, config.early_stop_callback],
                  logger=logger,
                  )
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)