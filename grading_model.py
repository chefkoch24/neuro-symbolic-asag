import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from scipy.stats import cosine
from sklearn import clone
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering, Adafactor, AutoTokenizer
from transformers import AdamW
from torch.optim import lr_scheduler
import logging

import config
import metrics
import sklearn
import myutils as utils
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
from torchmetrics import Accuracy, F1Score, Precision, Recall
from paraphrase_scorer import ParaphraseScorerSBERT


from model import TokenClassificationModel


class GradingModelClassification(LightningModule):
    def __init__(self, checkpoint: str, symbolic_learner, model_name, rubrics):
        super().__init__()
        self.save_hyperparameters()
        self.model = TokenClassificationModel(config.MODEL_NAME).load_from_checkpoint(checkpoint) #("/path/to/checkpoint.ckpt")
        self.loss = nn.CrossEntropyLoss()
        self.rubrics = rubrics
        self.symbolic_models = self.__init_learners__(symbolic_learner)
        self.para_detector = ParaphraseScorerSBERT()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        #metrics
        self.f1 = F1Score(num_classes=3, average='micro')
        self.accuracy = Accuracy()

    def __init_learners__(self, symbolic_learner):
            symbolic_models = {}
            for qid in list(self.rubrics.keys()):
                symbolic_models[qid] = clone(symbolic_learner)
            return symbolic_models


    def forward(self, input_ids, attention_mask, qid):
        outputs = self.model(input_ids, attention_mask=attention_mask)[0]
        predictions = torch.argmax(outputs, dim=-1)
        justification_cues = self.get_justification_cues(predictions)
        scoring_vectors = self.get_scoring_vectors(justification_cues, input_ids, question_id=qid)
        return scoring_vectors

    def get_justification_cues(self, predictions):
        justification_cues = []
        for p in predictions:
            true_predictions = metrics.silver2target(p)
            spans = metrics.get_spans_from_labels(true_predictions)
            justification_cues.append(spans)
        return justification_cues

    def get_scoring_vectors(self, justification_cues, input_ids, question_id):
        # get batched data
        rubric = self.rubrics[question_id] #expects the same question_id
        scoring_vectors = []
        for jus_cue, input_id in zip(justification_cues, input_ids):
            qid = jus_cue['qid']
            cue = jus_cue['text']
            student_answer = self.tokenizer.decode(input_id)
            max_idxs, max_vals = [], []
            for span in jus_cue:
                span_text = student_answer[span[0]:span[1]]
                sim = self.para_detector.detect_paraphrases(span_text, rubric)
                max_idxs.append(np.argmax(sim))
                max_vals.append(np.max(sim))
            # make sure that the absolute maximum is taken
            scoring_vector = np.zeros((len(self.rubrics[qid])))
            for mi, mv in zip(max_idxs, max_vals):
                if scoring_vector[mi] < mv:
                    scoring_vector[mi] = mv
            scoring_vectors.append(scoring_vector)
            return scoring_vectors


    def training_step(self, batch, batch_idx):
        qid = batch['question_id'][0]
        scoring_vectors = self.forward(batch['input_ids'], batch['attention_mask'], qid)
        self.symbolic_models[qid].fit(scoring_vectors, batch['labels'])
        y_pred = self.symbolic_models[qid].predict(scoring_vectors)
        loss = self.loss(y_pred, batch['class'])
        return loss

    def validation_step(self, batch, batch_idx):
        qid = batch['qid']
        scoring_vectors = self.forward(batch['input_ids'], batch['attention_mask'], qid)
        y_pred = self.symbolic_models[qid].predict(scoring_vectors)
        loss = self.loss_function(y_pred, batch['class'])
        batch['prediction'] = y_pred
        batch['scoring_vectors'] = scoring_vectors
        return batch


    def validation_epoch_end(self, outputs):
        acc = self.accuracy(outputs['prediction'], outputs['class'])
        f1 = self.f1(outputs['prediction'], outputs['class'])
        self.log('val_acc', acc)
        self.log('val_f1', f1)

    def configure_optimizers(self):
        # optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=self.eps, betas=self.betas, weight_decay=self.weight_decay)
        optimizer = Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer

class GradingModelTrivial():
    def __init__(self, max_score = 1, th=0.5):
        self.max_score = max_score
        self.th = th

    #TODO: rounding on a range?
    #TODO: think about if scoring vector is the right input

    def forward(self, scoring_vector, rubric):
        score = 0
        for s, r in zip(scoring_vector, rubric):
            if s > self.th:
                score += r['points']
        if score >= self.max_score:
            score = self.max_score
            classification = 0 #correct
        elif score > 0:
            classification = 1 #partially correct
        else:
            classification = 2 #incorrect

        return score, classification