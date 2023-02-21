# model class
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss, NLLLoss, BCEWithLogitsLoss, KLDivLoss
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering, Adafactor
from transformers import AdamW
from torch.optim import lr_scheduler
import metrics

# Model
class TokenClassificationModel(LightningModule):
    def __init__(self, model_name: str, rubrics=None, lr=0.001, eps=1e-08, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=0, ):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.loss = nn.CrossEntropyLoss()
        #self.loss = NLLLoss()
        #self.loss = nn.BCEWithLogitsLoss()
        #self.loss = nn.KLDivLoss() #reduction='none' for attention mask
        #kl loss and cross entropy loss are working with probabilities in labels and logits
        self.rubrics = rubrics
        #Hyperparameters
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.betas = betas


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
        metric = metrics.compute_metrics(outputs)
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True)
        return metric

    def test_step(self, batch, batch_idx):
        data = batch
        logits, loss = self.forward(data['input_ids'], data['attention_mask'], data['labels'])
        data['logits'] = logits
        data['loss'] = loss
        return data

    def test_epoch_end(self, outputs):
        metric = metrics.compute_metrics(outputs)
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True)
        return metric

    def configure_optimizers(self):
        #optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=self.eps, betas=self.betas, weight_decay=self.weight_decay)
        optimizer = Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer
