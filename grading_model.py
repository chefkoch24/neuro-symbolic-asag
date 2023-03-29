import os

import numpy as np
import torch
import torch.nn as nn
from incremental_trees.models.regression.streaming_rfr import StreamingRFR
from joblib import dump
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AdamW
import metrics
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
from paraphrase_scorer import BertScorer
import myutils as utils

from justification_cue_model import *


class Summation:
    def __init__(self, rubric, mode='regression', th=0.8, num_classes=3):
        self.rubric = rubric
        self.th = th
        self.mode = mode
        self.num_classes = num_classes

    def predict(self, scoring_vectors):
        y_preds_regression = []
        for scoring_vector in scoring_vectors:
            analytical_points = np.array(self.rubric['points'].tolist())
            scoring_vector = np.array([1 if s >= self.th else 0 for s in scoring_vector])
            score = sum(analytical_points * scoring_vector)
            y_preds_regression.append(float(score))
        return np.array(y_preds_regression)


    def predict_proba(self, scoring_vectors):
        y_preds_classification = []
        for scoring_vector in scoring_vectors:
            analytical_points = np.array(self.rubric['points'].tolist())
            scoring_vector = np.array([1 if s >= self.th else 0 for s in scoring_vector])
            score = sum(analytical_points * scoring_vector)
            if score >= 1:  # correct
                classification = [1.0, 0.0, 0.0]
            elif score < 1 and score > 0:  # partially correct
                classification = [0.0, 1.0, 0.0]
            else:  # incorrect
                classification = [0.0, 0.0, 1.0]
            y_preds_classification.append(classification)
        return np.array(y_preds_classification)


class GradingModel(LightningModule):
    def __init__(self, checkpoint: str, rubrics, model_name, mode='classification', learning_strategy='decision_tree',
                 task='token_classification', symbolic_models=None, lr=0.1, summation_th=0.8):
        super().__init__()
        self.task = task
        if self.task == 'token_classification':
            self.model = TokenClassificationModel.load_from_checkpoint(checkpoint)#, map_location=self.device)
        elif self.task == 'span_prediction':
            self.model = SpanPredictionModel.load_from_checkpoint(checkpoint)#, map_location=self.device)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.rubrics = rubrics
        self.mode = mode
        self.learning_strategy = learning_strategy
        self.classes = np.array([0, 1, 2])
        self.epoch = 1
        self.summation_th = summation_th
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.init_scoring_vector_logging()
        self.symbolic_models = self._init_symbolic_models(symbolic_models)
        self.para_detector = BertScorer()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, token_type_ids, question_ids, return_justification_cues=False):
        # only use batch
        q_id = question_ids[0]
        if self.task == 'token_classification':
            true_predictions, true_input_ids = self.get_token_classification_output(input_ids, attention_mask, token_type_ids, question_ids)
            justification_cues = self.get_justification_cues(true_predictions)
        elif self.task == 'span_prediction':
            justification_cues,true_input_ids = self.get_span_classification_output(input_ids, attention_mask, token_type_ids, question_ids)

        scoring_vectors = self.get_scoring_vectors(justification_cues, true_input_ids, question_ids)
        self.scoring_vectors_logging[q_id].append(scoring_vectors.tolist())
        if return_justification_cues:
            return scoring_vectors, justification_cues
        else:
            return scoring_vectors

    def get_token_classification_output(self, input_ids, attention_mask, token_type_ids, question_ids):
        outputs = self.model(input_ids=input_ids.squeeze(1),
                             attention_mask=attention_mask.squeeze(1))  # remove one unnecessary dimension
        predictions = torch.argmax(outputs, dim=-1)
        attention_mask = attention_mask.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        input_ids = input_ids.squeeze(1)
        # logic for setting up that only the inputs from the student answer are used
        true_predictions = [[predictions[i][j].item() for j in range(len(predictions[i])) if
                             token_type_ids[i][j].item() == 0 and attention_mask[i][j].item() == 1] for i in
                            range(len(predictions))]
        true_input_ids = [[input_ids[i][j].item() for j in range(len(predictions[i])) if
                           token_type_ids[i][j].item() == 0 and attention_mask[i][j].item() == 1] for i in
                          range(len(predictions))]
        # remove special tokens
        true_predictions = [x[1:-1] for x in true_predictions]
        true_input_ids = [x[1:-1] for x in true_input_ids]
        return true_predictions, true_input_ids

    def get_span_classification_output(self, input_ids, attention_mask, token_type_ids, question_ids):
        # for every element in the batch
        q_id = question_ids[0]
        justification_cues = []
        true_input_ids = []
        for input in input_ids:
            jus_cues= []
            student_answer = self.tokenizer.batch_decode(input, skip_special_tokens=True)[0]
            for re in self.rubrics[q_id]['key_element']:
                input = self.tokenizer(re,student_answer, truncation=True, return_tensors='pt', return_token_type_ids=True, max_length=512)
                i_ids = input['input_ids'].to(self.device)
                a_mask = input['attention_mask'].to(self.device)
                t_ids = input['token_type_ids'].to(self.device)
                start_logits, end_logits = self.model(i_ids, attention_mask=a_mask)
                mask = (a_mask == 1) & (t_ids == 1)
                start_logits_masked = start_logits.masked_fill(~mask, float('-inf'))
                end_logits_masked = end_logits.masked_fill(~mask, float('-inf'))

                start_predictions = start_logits_masked.argmax(dim=-1)
                end_predictions = end_logits_masked.argmax(dim=-1)
                # reverse the offset
                tokenized_len = len(self.tokenizer.tokenize(re)) #not return special tokens without the flag
                offset = tokenized_len + 2
                jus_cues.append((start_predictions.item()-offset, end_predictions.item()-offset))
            justification_cues.append(jus_cues)
            true_input_ids.append(input['input_ids'][0].tolist()[1:-1])
        return justification_cues, true_input_ids

    def _init_symbolic_models(self, symbolic_models):
        if symbolic_models is None:
            symbolic_models = {}
            for qid in list(self.rubrics.keys()):
                max_features = len(self.rubrics[qid]['key_element'].tolist())
                if self.mode == 'classification':
                    if self.learning_strategy == 'decision_tree':
                        symbolic_models[qid] = StreamingRFC(n_estimators_per_chunk=1, max_features=max_features)
                    else:
                        symbolic_models[qid] = Summation(self.rubrics[qid], mode='classification', th=self.summation_th,
                                                         num_classes=len(self.classes))
                elif self.mode == 'regression':

                    if self.learning_strategy == 'decision_tree':
                        symbolic_models[qid] = StreamingRFR(n_estimators_per_chunk=1, max_features=max_features)
                    else:
                        symbolic_models[qid] = Summation(self.rubrics[qid], mode='regression', th=self.summation_th, num_classes=3)
        else:
            self.symbolic_models = symbolic_models
        return symbolic_models

    def get_justification_cues(self, predictions):
        justification_cues = []
        for p in predictions:
            spans = metrics.get_spans_from_labels(p)
            justification_cues.append(spans)
        return justification_cues

    def get_scoring_vectors(self, justification_cues, input_ids, question_ids):
        # get batched data
        qid = question_ids[0]
        rubric = self.rubrics[qid]  # expects the same question_id
        scoring_vectors = []
        for jus_cue, input_id in zip(justification_cues, input_ids):
            # fuzzy matching
            scoring_vector = [0] * len(rubric['key_element'])
            for span in jus_cue:
                cue_text = self.tokenizer.decode(input_id[span[0]:span[1]], skip_special_tokens=True)
                sim = self.para_detector.detect_score_key_elements(cue_text, rubric)
                for i, s in enumerate(sim):
                    if s > scoring_vector[i]:
                        scoring_vector[i] = s
            scoring_vectors.append(scoring_vector)
        return np.array(scoring_vectors)

    def init_scoring_vector_logging(self):
        self.scoring_vectors_logging = {}
        for k in self.rubrics.keys():
            self.scoring_vectors_logging[k] = []

    def training_step(self, batch, batch_idx):
        q_id = batch['question_id'][0]
        symbolic_model = self.symbolic_models[q_id]
        if self.mode == 'classification':
            labels = batch['class']
            loss_function = self.cross_entropy_loss
            prediction_function = symbolic_model.predict_proba
        elif self.mode == 'regression':
            labels = batch['score']
            loss_function = self.mse_loss
            prediction_function = symbolic_model.predict  # independent if decision tree or summation

        scoring_vectors = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                                       batch['question_id'])

        labels_cpu = torch.clone(labels).cpu().detach().numpy()
        if self.learning_strategy == 'decision_tree':
            self.symbolic_models[q_id] = symbolic_model.partial_fit(scoring_vectors, labels_cpu,
                                                                    classes=self.classes)
        y_pred = prediction_function(scoring_vectors)
        if self.mode == 'regression':
            y_pred = [utils.scaled_rounding(y) for y in y_pred]
        y_pred = torch.tensor(y_pred, requires_grad=True, device=self.device)
        loss = loss_function(y_pred, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        q_id = batch['question_id'][0]
        symbolic_model = self.symbolic_models[q_id]
        if self.mode == 'classification':
            labels = batch['class']
            loss_function = self.cross_entropy_loss
            prediction_function = symbolic_model.predict_proba
        elif self.mode == 'regression':
            labels = batch['score']
            loss_function = self.mse_loss
            prediction_function = symbolic_model.predict
        scoring_vectors = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                                       batch['question_id'])
        y_pred = prediction_function(scoring_vectors)
        if self.mode == 'regression':
            y_pred = [utils.scaled_rounding(y) for y in y_pred]
        y_pred = torch.tensor(y_pred, requires_grad=True, device=self.device)
        loss = loss_function(y_pred, labels)
        batch['prediction'] = y_pred
        batch['loss'] = loss
        return batch

    def validation_epoch_end(self, outputs):
        metric = {}
        if self.mode == 'classification':
            metric = metrics.compute_grading_classification_metrics(outputs)
        elif self.mode == 'regression':
            metric = metrics.compute_grading_regression_metrics(outputs)
        self.log_dict(metric)
        # save scoring vectors
        EXPERIMENT_NAME = utils.get_experiment_name(['grading', self.task, self.mode, self.learning_strategy])
        utils.save_json(self.scoring_vectors_logging, 'logs/' + EXPERIMENT_NAME + '/scoring_vectors', 'scoring_vectors_epoch_{}.json'.format(self.epoch))
        self.init_scoring_vector_logging()
        # save symbolic models
        self._save_symbolic_models()
        self.epoch += 1
        # save the incremental models

        return metric

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def predict_step(self, batch, batch_idx):
        # used in the trainer.predict() method
        q_id = batch['question_id'][0]
        symbolic_model = self.symbolic_models[q_id]
        if self.mode == 'classification':
            prediction_function = symbolic_model.predict_proba
        elif self.mode == 'regression':
            prediction_function = symbolic_model.predict
        scoring_vectors = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                                       batch['question_id'])
        y_pred = prediction_function(scoring_vectors)
        if self.mode == 'regression':
            y_pred = [utils.scaled_rounding(y) for y in y_pred]
        return y_pred

    def predict(self, batch, batch_idx, return_reasoning=False):
        # custom prediction method used for explainable inference
        if return_reasoning:
            scoring_vectors, justification_cues = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                                           batch['question_id'], return_justification_cues=return_reasoning)
            q_id = batch['question_id'][0]
            symbolic_model = self.symbolic_models[q_id]
            if self.mode == 'classification':
                prediction_function = symbolic_model.predict_proba
            elif self.mode == 'regression':
                prediction_function = symbolic_model.predict
            y_pred = prediction_function(scoring_vectors)
            if self.mode == 'regression':
                y_pred = [utils.scaled_rounding(y) for y in y_pred]
            return y_pred, scoring_vectors, justification_cues
        else:
            return self.predict_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        # optimizer = Adafactor(self.model.parameters(), lr=None, relative_step=True)
        return optimizer

    @classmethod
    def load_from_checkpoint(self, checkpoint_path, map_location=None, hparams_file=None, strict=True, symbolic_models=None, **kwargs):
        m = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, **kwargs)
        m.symbolic_models = symbolic_models
        return m

    def _save_symbolic_models(self):
        EXPERIMENT_NAME = utils.get_experiment_name(['grading', self.task, self.mode, self.learning_strategy])
        path = 'logs/' + EXPERIMENT_NAME + '/symbolic_models/epoch_' + str(self.epoch)
        if not os.path.exists(path):
            os.makedirs(path)
        for qid, dt in self.symbolic_models.items():
            dump(dt, path + '/' + qid + '.joblib')
