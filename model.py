# model class
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import AutoModelForTokenClassification
from transformers import AdamW
from torch.optim import lr_scheduler
import logging

logging.basicConfig(level=logging.ERROR)


# Model
class TokenClassificationModel(LightningModule):
    def __init__(self, model_name: str, rubrics):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.loss = nn.CrossEntropyLoss()
        self.rubrics = rubrics

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)[0]
        if labels is not None:
            if labels.size()[0] > 1:
                labels = labels.view(-1)
            loss = self.loss(outputs.view(-1, outputs.size(-1)), labels)
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
        print(loss)
        # loss = self.loss(outputs.view(-1, outputs.size(-1)), labels)
        return {'logits': logits,
                'labels': data['labels'],
                'input_ids': data['input_ids'],
                # [self.tokenizer.decode(x, skip_special_tokens=True) for x in data['input_ids']],
                'class': data['label'],
                'question_id': data['question_id'],
                'loss': loss
                }

    def validation_epoch_end(self, outputs):
        metrics = dict() #compute_metrics(outputs)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics['val_loss'] = avg_loss
        print(metrics)
        return metrics  # metrics['val_loss'] = avg_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]