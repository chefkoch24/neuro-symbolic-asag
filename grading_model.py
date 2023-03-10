import numpy as np
import torch
import torch.nn as nn
from incremental_trees.models.regression.streaming_rfr import StreamingRFR
from pytorch_lightning import LightningModule
from transformers import Adafactor, AutoTokenizer, AdamW
from torch.optim import lr_scheduler
import metrics
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
from torchmetrics import Accuracy, F1Score, Precision, Recall
from paraphrase_scorer import BertScorer

from model import TokenClassificationModel


class GradingModelTrivial(LightningModule):
    def __init__(self, checkpoint: str, model_name, rubrics, th=0.5):
        super().__init__()
        self.model = TokenClassificationModel.load_from_checkpoint(checkpoint) #("/path/to/checkpoint.ckpt")
        self.loss = nn.CrossEntropyLoss()
        self.rubrics = rubrics
        self.para_detector = BertScorer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classes = [0,1,2] # 0 = correct, 1 = partially correct, 2 = incorrect
        self.f1_score = F1Score(task='multiclass', num_classes=3, average='none')
        self.accuracy = Accuracy(task='multiclass', num_classes=3, average='none')
        self.prec = Precision(task='multiclass', num_classes=3, average='none')
        self.recall = Recall(task='multiclass', num_classes=3, average='none')
        self.th = th
        self.save_hyperparameters()


    def forward(self, input_ids, attention_mask, question_ids, labels=None):
        outputs = self.model(input_ids=input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1)) #remove one unnecessary dimension
        predictions = torch.argmax(outputs, dim=-1)
        true_predictions, true_input_ids = [], []
        attention_mask = attention_mask.squeeze(1)
        input_ids = input_ids.squeeze(1)
        for i in range(len(predictions)):
            temp, inp= [], []
            for j in range(len(predictions[i])):
                if attention_mask[i][j].item() == 1:
                    inp.append(input_ids[i][j].item())
                    temp.append(predictions[i][j].item())
            true_predictions.append(temp)
            true_input_ids.append(inp)
        justification_cues = self.get_justification_cues(true_predictions)
        scoring_vectors = self.get_scoring_vectors(justification_cues, true_input_ids, question_ids)
        y_pred = self.predict_class(scoring_vectors, question_ids)
        if labels is not None:
            loss = self.loss(torch.tensor(y_pred, requires_grad=True), labels)
            return torch.tensor(y_pred), loss
        return torch.tensor(y_pred)

    def predict_class(self, scoring_vectors, question_ids):

        y_preds = []
        for scoring_vector, qid in zip(scoring_vectors, question_ids):
            analytical_points = np.array(self.rubrics[qid]['points'].tolist())
            scoring_vector = np.array([1 if s >= self.th else 0 for s in scoring_vector])
            score = sum(analytical_points * scoring_vector)
            if score >= 1: # correct
                classification = 0
            elif score < 1 and score > 0: # partially correct
                classification = 1
            else: # incorrect
                classification = 2
            y_preds.append(classification)
        one_hot = torch.nn.functional.one_hot(torch.tensor(y_preds), num_classes=len(self.classes))
        return one_hot.float()


    def get_justification_cues(self, predictions):
        justification_cues = []
        for p in predictions:
            spans = metrics.get_spans_from_labels(p)
            justification_cues.append(spans)
        return justification_cues

    def get_scoring_vectors(self, justification_cues, input_ids, question_ids):
        scoring_vectors = []
        for jus_cue, input_id, qid in zip(justification_cues, input_ids, question_ids, ):
            rubric = self.rubrics[qid]  # expects the same question_id
            max_idxs, max_vals = [], []
            for span in jus_cue:
                cue_text = self.tokenizer.decode(input_id[span[0]:span[1]], skip_special_tokens=True)
                sim = self.para_detector.detect_score_key_elements(cue_text, rubric)
                max_idxs.append(np.argmax(sim))
                max_vals.append(np.max(sim))
            # make sure that the absolute maximum is taken
            scoring_vector = np.zeros((len(self.rubrics[qid])))
            for mi, mv in zip(max_idxs, max_vals):
                if scoring_vector[mi] < mv:
                    scoring_vector[mi] = mv
            scoring_vectors.append(scoring_vector)
        return np.array(scoring_vectors)


    def training_step(self, batch, batch_idx):
        _, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['question_id'], batch['class'])
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['question_id'], batch['class'])
        batch['prediction'] = y_pred
        batch['loss'] = loss
        return batch


    def validation_epoch_end(self, outputs):
        predictions = torch.cat([x['prediction'] for x in outputs])
        labels = torch.cat([x['class'] for x in outputs])
        f1 = self.f1_score(predictions, labels)
        acc = self.accuracy(predictions, labels)
        precision = self.prec(predictions, labels)
        recall = self.recall(predictions, labels)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        compare_metrics = metrics.compute_grading_classification_metrics(outputs)
        self.log('val_loss', avg_loss)
        self.log('val_f1_correct', f1[0].item())
        self.log('val_f1_partial', f1[1].item())
        self.log('val_f1_incorrect', f1[2].item())
        self.log('val_acc_correct', acc[0].item())
        self.log('val_acc_partial', acc[1].item())
        self.log('val_acc_incorrect', acc[2].item())
        self.log('val_precision_correct', precision[0].item())
        self.log('val_precision_partial', precision[1].item())
        self.log('val_precision_incorrect', precision[2].item())
        self.log('val_recall_correct', recall[0].item())
        self.log('val_recall_partial', recall[1].item())
        self.log('val_recall_incorrect', recall[2].item())
        self.log_dict(compare_metrics, prog_bar=False)
        return avg_loss

    def configure_optimizers(self):
        # optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=self.eps, betas=self.betas, weight_decay=self.weight_decay)
        optimizer = Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer


class GradingModel(LightningModule):
    def __init__(self, checkpoint: str, model_name, rubrics, mode='classification'):
        super().__init__()
        self.model = TokenClassificationModel.load_from_checkpoint(checkpoint) #("/path/to/checkpoint.ckpt")
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.rubrics = rubrics
        self.mode = mode
        self.symbolic_models = self.__init_learners__()
        self.para_detector = BertScorer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classes = np.array([0,1,2])
        self.save_hyperparameters()


    def forward(self, input_ids, attention_mask, question_ids, token_type_ids, labels=None):
        q_id = question_ids[0]
        outputs = self.model(input_ids=input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1)) #remove one unnecessary dimension
        predictions = torch.argmax(outputs, dim=-1)
        true_predictions, true_input_ids = [], []
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
        justification_cues = self.get_justification_cues(true_predictions)
        scoring_vectors = self.get_scoring_vectors(justification_cues, true_input_ids, question_ids)
        symbolic_model = self.symbolic_models[q_id]
        self.symbolic_models[q_id] = symbolic_model.partial_fit(scoring_vectors,labels.detach().numpy(),
                                                                self.classes)
        if self.mode == 'classification':
            y_pred = self.symbolic_models[q_id].predict_proba(scoring_vectors)
        elif self.mode == 'regression':
            y_pred = self.symbolic_models[q_id].predict(scoring_vectors)
        if labels is not None:
            if self.mode == 'classification':
                loss = self.cross_entropy_loss(torch.tensor(y_pred, requires_grad=True), labels)
            elif self.mode == 'regression':
                loss = self.mse_loss(torch.tensor(y_pred, requires_grad=True), labels)
            return torch.tensor(y_pred), loss
        return torch.tensor(y_pred)

    def __init_learners__(self):
        symbolic_models = {}
        for qid in list(self.rubrics.keys()):
            if self.mode == 'classification':
                symbolic_models[qid] = StreamingRFC()
            elif self.mode == 'regression':
                symbolic_models[qid] = StreamingRFR()
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
        rubric = self.rubrics[qid] #expects the same question_id
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


    def training_step(self, batch, batch_idx):
        if self.mode == 'classification':
            _, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['question_id'], batch['token_type_ids'], batch['class'])
        elif self.mode == 'regression':
            _, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['question_id'], batch['token_type_ids'], batch['score'])
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == 'classification':
            y_pred, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['question_id'], batch['token_type_ids'], batch['class'])
        elif self.mode == 'regression':
            y_pred, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['question_id'], batch['token_type_ids'], batch['score'])
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

    def configure_optimizers(self):
        # optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=self.eps, betas=self.betas, weight_decay=self.weight_decay)
        optimizer = Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer