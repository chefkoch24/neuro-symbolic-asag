# model class
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering, Adafactor, \
    get_constant_schedule_with_warmup, AdamW
import metrics


# Model
class TokenClassificationModel(LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()


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
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        data = batch
        logits, loss = self.forward(data['input_ids'], data['attention_mask'], data['labels'])
        data['logits'] = logits
        data['loss'] = loss
        return data


    def validation_epoch_end(self, outputs):
        metric = metrics.compute_metrics_token_classification(outputs)
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True)
        return metric

    def test_step(self, batch, batch_idx):
        data = batch
        logits, loss = self.forward(data['input_ids'], data['attention_mask'], data['labels'])
        data['logits'] = logits
        data['loss'] = loss
        return data

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'])
        predictions = torch.argmax(outputs, dim=-1)
        return predictions

    def test_epoch_end(self, outputs):
        metric = metrics.compute_metrics_token_classification(outputs)
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True)
        return metric

    def configure_optimizers(self):
        #optimizer = AdamW(self.model.parameters(), lr=self.lr)
        # replace AdamW with Adafactor
        optimizer = Adafactor(self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return optimizer #get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps)


# Model
class SpanPredictionModel(LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, return_dict=True)
        if start_positions is not None and end_positions is not None:
            return outputs.start_logits, outputs.end_logits, outputs.loss
        return outputs.start_logits, outputs.end_logits

    def training_step(self, batch, batch_idx):
        _, _, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['start_positions'], batch['end_positions'])
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        start_logits, end_logits, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['start_positions'], batch['end_positions'])
        batch['start_logits'] = start_logits
        batch['end_logits'] = end_logits
        batch['loss'] = loss
        return batch


    def validation_epoch_end(self, outputs):
        metric = metrics.compute_metrics_span_prediction(outputs)
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True)
        return metric

    def test_step(self, batch, batch_idx):
        start_logits, end_logits, loss = self.forward(batch['input_ids'], batch['attention_mask'], batch['start_positions'], batch['end_positions'])
        batch['start_logits'] = start_logits
        batch['end_logits'] = end_logits
        batch['loss'] = loss
        return batch

    def test_epoch_end(self, outputs):
        metric = metrics.compute_metrics_span_prediction(outputs)
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True)
        return metric

    def predict_step(self, batch, batch_idx):
        start_logits, end_logits = self.forward(batch['input_ids'], batch['attention_mask'])
        mask = (batch['attention_mask'] == 1) & (batch['token_type_ids'] == 1)
        start_logits_masked = start_logits.masked_fill(~mask, float('-inf'))
        end_logits_masked = end_logits.masked_fill(~mask, float('-inf'))

        start_predictions = start_logits_masked.argmax(dim=-1)
        end_predictions = end_logits_masked.argmax(dim=-1)
        return torch.stack((start_predictions, end_predictions))

    def configure_optimizers(self):
        #lr_scheduler=get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps)
        #ptimizer = AdamW(self.model.parameters(), lr=0.001)
        optimizer = Adafactor(self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return optimizer #, lr_scheduler
