import os
from collections import defaultdict
import numpy as np
from joblib import dump
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from transformers import AutoTokenizer, AdamW
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
        y_preds_classification = []
        for scoring_vector in scoring_vectors:
            analytical_points = np.array(self.rubric['points'].tolist())
            scoring_vector = np.array([1 if s >= self.th else 0 for s in scoring_vector])
            score = sum(analytical_points * scoring_vector)
            if score >= 1:  # correct
                classification = 0
            elif score < 1 and score > 0:  # partially correct
                classification = 1
            else:  # incorrect
                classification = 2
            y_preds_regression.append(float(score))
            y_preds_classification.append(classification)
        if self.mode == 'regression':
            y_pred = np.array(y_preds_regression)
        elif self.mode == 'classification':
            y_pred = np.array(y_preds_classification)
        return y_pred


class GradingModel(LightningModule):
    def __init__(self, checkpoint: str, rubrics, model_name, mode='classification', learning_strategy='decision_tree',
                 task='token_classification', symbolic_models=None, lr=0.001, summation_th=0.8, matching='exact', is_fixed_learner=True, experiment_name=''):
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
        self.epoch = 0
        self.summation_th = summation_th
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # for saving and the training step of the symbolic models
        self.all_targets = defaultdict(list)
        self.all_train_scoring_vectors = defaultdict(list)
        self.all_test_scoring_vectors = defaultdict(list)
        self.symbolic_models = self._init_symbolic_models(symbolic_models)
        self.para_detector = BertScorer()
        self.lr = lr
        self.is_fixed_learner = is_fixed_learner
        self.matching = matching
        self.experiment_name = experiment_name
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, token_type_ids, question_ids, return_justification_cues=False):
        # only use batch
        q_id = question_ids[0]
        if self.task == 'token_classification':
            true_predictions, true_input_ids = self.get_token_classification_output(input_ids, attention_mask, token_type_ids, question_ids)
            justification_cues = self.get_justification_cues(true_predictions)
        elif self.task == 'span_prediction':
            justification_cues,true_input_ids = self.get_span_classification_output(input_ids, attention_mask, token_type_ids, question_ids)


        if return_justification_cues:
            scoring_vectors, justification_cue_texts = self.get_scoring_vectors(justification_cues, true_input_ids, question_ids, return_texts=True)
            return scoring_vectors, justification_cues, justification_cue_texts
        else:
            scoring_vectors = self.get_scoring_vectors(justification_cues, true_input_ids,question_ids)
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
                        symbolic_models[qid] = DecisionTreeClassifier(max_features=max_features, max_depth=max_features)
                    else:
                        symbolic_models[qid] = Summation(self.rubrics[qid], mode='classification', th=self.summation_th,
                                                         num_classes=len(self.classes))
                elif self.mode == 'regression':

                    if self.learning_strategy == 'decision_tree':
                        symbolic_models[qid] = DecisionTreeRegressor(max_features=max_features, max_depth=max_features)
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

    def get_scoring_vectors(self, justification_cues, input_ids, question_ids, return_texts=False):
        cue_texts = []
        # get batched data
        qid = question_ids[0]
        scoring_vectors = []
        for jus_cue, input_id in zip(justification_cues, input_ids):
            # choose between hard or fuzzy matching
            if self.matching == 'exact':
                scoring_vector, cue_text = self.hard_matching(jus_cue, input_id, qid)
            elif self.matching == 'fuzzy':
                scoring_vector, cue_text = self.fuzzy_matching(jus_cue, input_id, qid)
            scoring_vectors.append(scoring_vector)
            cue_texts.append(cue_text)
        if return_texts:
            return scoring_vectors, cue_texts
        else:
            return scoring_vectors

    def hard_matching(self, justification_cue, input_id, question_id):
        # hard matching one justification cue per key element
        rubric = self.rubrics[question_id]
        max_idxs, max_vals = [], []
        cue_texts = []
        for span in justification_cue:
            cue_text = self.tokenizer.decode(input_id[span[0]:span[1]], skip_special_tokens=True)
            cue_texts.append(cue_text)
            sim = self.para_detector.detect_score_key_elements(cue_text, rubric)
            max_idxs.append(np.argmax(sim))
            max_vals.append(np.max(sim))
        # make sure that the absolute maximum is taken
        scoring_vector = np.zeros((len(self.rubrics[question_id])))
        for mi, mv in zip(max_idxs, max_vals):
            if scoring_vector[mi] < mv:
                scoring_vector[mi] = mv
        return scoring_vector.tolist(), cue_texts


    def fuzzy_matching(self, justification_cue, input_id, question_id):
        #fuzzy matching
        rubric = self.rubrics[question_id]
        scoring_vector = [0] * len(rubric['key_element'])
        cue_texts = []
        for span in justification_cue:
            cue_text = self.tokenizer.decode(input_id[span[0]:span[1]], skip_special_tokens=True)
            cue_texts.append(cue_text)
            sim = self.para_detector.detect_score_key_elements(cue_text, rubric)
            for i, s in enumerate(sim):
               if s > scoring_vector[i]:
                   scoring_vector[i] = s
        return scoring_vector, cue_texts


    def training_step(self, batch, batch_idx):
        q_id = batch['question_id'][0]
        symbolic_model = self.symbolic_models[q_id]
        prediction_function = symbolic_model.predict  # independent if decision tree or summation
        if self.mode == 'classification':
            labels = batch['class']
            loss_function = self.cross_entropy_loss
        elif self.mode == 'regression':
            labels = batch['score']
            loss_function = self.mse_loss
        scoring_vectors = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                                       batch['question_id'])
        labels_cpu = torch.clone(labels).cpu().detach().numpy()
        self.all_targets[q_id].append(labels_cpu)
        self.all_train_scoring_vectors[q_id].append(scoring_vectors)
        loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        if self.epoch > 0:
           loss, y_pred = self.forward_step(scoring_vectors, labels, prediction_function, loss_function)
        return loss

    def forward_step(self, scoring_vectors, labels, prediction_function, loss_function):
        y_pred = prediction_function(scoring_vectors)
        if self.mode == 'regression':
            y_pred = [utils.scaled_rounding(y) for y in y_pred]
            y_pred = torch.tensor(y_pred, requires_grad=True, device=self.device)
        elif self.mode == 'classification':
            y_pred = [int(y) for y in y_pred]  # because one_hot requires index indices
            y_pred = torch.nn.functional.one_hot(torch.tensor(y_pred, device=self.device), num_classes=len(self.classes)).float().requires_grad_(True)
        loss = loss_function(y_pred, labels)
        return loss, y_pred


    def training_epoch_end(self, outputs):
        # fit only in the first epoch
        if self.learning_strategy == 'decision_tree':
            for qid in self.rubrics.keys():
                if self.all_train_scoring_vectors[qid] != []:
                    self.all_train_scoring_vectors[qid] = utils.flat_list(self.all_train_scoring_vectors[qid])
                    self.all_test_scoring_vectors[qid] = utils.flat_list(self.all_test_scoring_vectors[qid])
                    self.all_targets[qid] = utils.flat_list(self.all_targets[qid])
                    if self.epoch == 0:
                        self.symbolic_models[qid] = self.symbolic_models[qid].fit(np.array(self.all_train_scoring_vectors[qid]), np.array(self.all_targets[qid]))
                    elif not self.is_fixed_learner:
                        self.symbolic_models[qid] = self.symbolic_models[qid].fit(np.array(self.all_train_scoring_vectors[qid]), np.array(self.all_targets[qid]))
        self._save_symbolic_models()
        # save scoring vectors
        utils.save_json(self.all_train_scoring_vectors, 'logs/' + self.experiment_name + '/scoring_vectors',
                        'train_scoring_vectors_epoch_{}.json'.format(self.epoch))
        utils.save_json(self.all_test_scoring_vectors, 'logs/' + self.experiment_name + '/scoring_vectors',
                        'test_scoring_vectors_epoch_{}.json'.format(self.epoch))
        # reset scoring vectors
        self.all_targets = defaultdict(list)
        self.all_train_scoring_vectors = defaultdict(list)
        self.all_test_scoring_vectors = defaultdict(list)
        self.epoch += 1


    def validation_step(self, batch, batch_idx):
        q_id = batch['question_id'][0]
        symbolic_model = self.symbolic_models[q_id]
        prediction_function = symbolic_model.predict
        if self.mode == 'classification':
            labels = batch['class']
            loss_function = self.cross_entropy_loss
        elif self.mode == 'regression':
            labels = batch['score']
            loss_function = self.mse_loss
        scoring_vectors = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                                       batch['question_id'])
        self.all_test_scoring_vectors[q_id].append(scoring_vectors)
        if self.epoch > 0:
            loss, y_pred = self.forward_step(scoring_vectors, labels, prediction_function, loss_function)
        else:
            # first epoch
            if self.mode == 'classification':
                y_pred = torch.tensor([[0.0, 0.0, 0.0]] * len(labels), requires_grad=True, device=self.device)
            elif self.mode == 'regression':
                y_pred = torch.tensor([0.0] * len(labels), requires_grad=True, device=self.device)
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
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

        return metric

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def predict_step(self, batch, batch_idx):
        # used in the trainer.predict() method
        q_id = batch['question_id'][0]
        symbolic_model = self.symbolic_models[q_id]
        prediction_function = symbolic_model.predict
        if self.mode == 'classification':
            labels = batch['class']
            loss_function = self.cross_entropy_loss
        elif self.mode == 'regression':
            labels = batch['score']
            loss_function = self.mse_loss
        scoring_vectors = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                                       batch['question_id'])
        loss, y_pred = self.forward_step(scoring_vectors, labels, prediction_function, loss_function)
        return y_pred


    def predict(self, batch, batch_idx, return_reasoning=False):
        # custom prediction method used for explainable inference
        if return_reasoning:
            q_id = batch['question_id'][0]
            symbolic_model = self.symbolic_models[q_id]
            prediction_function = symbolic_model.predict
            if self.mode == 'classification':
                labels = batch['class']
                loss_function = self.cross_entropy_loss
            elif self.mode == 'regression':
                labels = batch['score']
                loss_function = self.mse_loss
            scoring_vectors, justification_cues, justification_cue_texts = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                                           batch['question_id'], return_justification_cues=True)
            loss, y_pred = self.forward_step(scoring_vectors, labels, prediction_function, loss_function)
            return y_pred, scoring_vectors, justification_cues, justification_cue_texts
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
        path = 'logs/' + self.experiment_name + '/symbolic_models/epoch_' + str(self.epoch)
        if not os.path.exists(path):
            os.makedirs(path)
        for qid, dt in self.symbolic_models.items():
            dump(dt, path + '/' + qid + '.joblib')
