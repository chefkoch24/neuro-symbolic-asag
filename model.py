# model class
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AdamW
from torch.optim import lr_scheduler
import logging
import metrics


# Model
class TokenClassificationModel(LightningModule):
    def __init__(self, model_name: str, rubrics=None):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.loss = nn.CrossEntropyLoss()

        self.rubrics = rubrics

    def create_labels(self, batch):
        targets = []
        for labels in batch:
            t = []
            for l in labels:
                if l > -100:
                    t.append([l, 1 - l])
                else:
                    t.append([l, l])
            targets.append(t)
        labels = torch.tensor(targets, dtype=torch.float64)
        return labels

    def remove_paddings(self, logits, target):
        new_target, new_logits = [], []
        for l, t in zip(logits, target):
            if t[0] != -100 and t[1] != -100:
                new_target.append(t)
                new_logits.append(l)

        target = torch.stack(new_target)
        logits = torch.stack(new_logits)
        return logits, target

    def loss_function(self, logits, labels):
        logits = logits.view(-1, 2)  # 2 - number of classes
        labels = self.create_labels(labels)
        labels = labels.view(-1, 2)
        logits, labels = self.remove_paddings(logits, labels)
        return self.loss(logits, labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)[0]
        if labels is not None:
            loss = self.loss_function(outputs, labels)
            return outputs, loss
        return outputs

    def training_step(self, batch, batch_idx):
        data = batch
        _, loss = self.forward(data['input_ids'], data['attention_mask'], data['labels'])
        logs = {'train_loss': loss}
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        logits, loss = self.forward(data['input_ids'], data['attention_mask'], data['labels'])
        # loss = self.loss(outputs.view(-1, outputs.size(-1)), labels)
        data['logits'] = logits
        data['loss'] = loss
        #{'logits': logits, 'loss': loss}.update(**data)
        return data


    def validation_epoch_end(self, outputs):
        metric = metrics.compute_metrics(outputs)
        #self.log(metrics=metric, prog_bar=True, logger=True)
        return metric  # metrics['val_loss'] = avg_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]

# Plain pytorch model
class SoftLabelTokenClassificationModel(torch.nn.Module):
    def __init__(self, model_name: str, rubrics):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.rubrics = rubrics
        self.loss = nn.CrossEntropyLoss()

    def create_labels(self, batch):
        targets = []
        for labels in batch:
            t = []
            for l in labels:
                if l > -100:
                    t.append([l, 1 - l])
                else:
                    t.append([l, l])
            targets.append(t)
        labels = torch.tensor(targets, dtype=torch.float64)
        return labels

    def remove_paddings(self, logits, target):
        new_target, new_logits = [], []
        for l, t in zip(logits, target):
            if t[0] != -100 and t[1] != -100:
                new_target.append(t)
                new_logits.append(l)

        target = torch.stack(new_target)
        logits = torch.stack(new_logits)
        return logits, target

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)[0]
        if labels is not None:
            outputs = outputs.view(-1, 2) #2 - number of classes
            labels = self.create_labels(labels)
            labels = labels.view(-1,2)
            outputs, labels = self.remove_paddings(outputs, labels)
            loss = self.loss(outputs, labels)
            return outputs, loss
        return outputs

class IterativeModel(LightningModule):
    def __init__(self, model_name: str, rubrics=None):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.loss_fct = CrossEntropyLoss()

    def loss_function(self, start_positions, end_positions, start_logits, end_logits):
        total_loss = None
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        start_loss = self.loss_fct(start_logits, start_positions)
        end_loss = self.loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        if start_positions is not None and end_positions is not None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                       end_positions=end_positions)[0]
            loss = self.loss_function(start_positions, end_positions, outputs[0], outputs[1])
            return outputs, loss
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        data = batch
        _, loss = self.forward(input_ids=data['input_ids'], attention_mask=data['attention_mask'], start_positions=data['start_positions'], end_positions=data['end_positions'])
        logs = {'train_loss': loss}
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        logits, loss = self.forward(input_ids=data['input_ids'], attention_mask=data['attention_mask'], start_positions=data['start_positions'], end_positions=data['end_positions'])
        # loss = self.loss(outputs.view(-1, outputs.size(-1)), labels)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]