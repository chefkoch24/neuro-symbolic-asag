# model class
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss, NLLLoss, BCEWithLogitsLoss, KLDivLoss
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
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

    def soft_cross_entropy_loss(self, input_logits, target_soft_labels, mask=None):
        # input_logits has shape [batch_size, num_classes, sequence_length]
        # target_soft_labels has shape [batch_size, num_classes, sequence_length]
        # mask has shape [batch_size, sequence_length] and is optional

        # First, convert the target soft labels to log-probabilities
        target_logprobs = torch.log(target_soft_labels + 1e-10)

        # Next, compute the KL divergence between the input logits and target log-probs

        loss = self.loss(torch.log_softmax(input_logits, dim=1), target_logprobs, )

        # If a mask is given, use it to mask out the padding tokens
        if mask is not None:
            loss = loss * mask.unsqueeze(1)

        # Finally, take the average loss over all non-padding tokens
        num_tokens = torch.sum(mask) if mask is not None else torch.numel(loss)
        loss = torch.sum(loss) / num_tokens

        return loss

    def create_labels(self, batch):
        targets = []
        for labels in batch:
            t = []
            for l in labels:
                if l > -100:
                    t.append([1-l, l])
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

    def remove_paddings_2(self, batch_logits, batch_targets):
        # create a mask where the values in the target tensor are not -100
        output_logits, output_targets = [], []
        for logits, targets in zip(batch_logits, batch_targets):
            mask = (targets != -100)
            new_logits = logits[mask]
            new_targets = targets[mask]
        output_logits.append(new_logits)
        output_targets.append(new_targets)

        # reshape the tensors to their original shapes
        output_logits = output_logits.reshape(batch_logits.shape[0], -1, batch_logits.shape[2])
        output_targets = output_targets.reshape(batch_targets.shape[0], -1, batch_targets.shape[2])

        return output_logits, output_targets

    def loss_function(self, logits, labels):
        logits = logits.view(-1, 2)  # 2 - number of classes
        labels = self.create_labels(labels)
        #logits = logits.transpose(1, 2)
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
        self.log('metrics', metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metric

    def test_step(self, batch, batch_idx):
        data = batch
        logits, loss = self.forward(data['input_ids'], data['attention_mask'], data['labels'])
        data['logits'] = logits
        data['loss'] = loss
        return data

    def test_epoch_end(self, outputs):
        metric = metrics.compute_metrics(outputs)
        self.log('metrics', metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metric

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, eps=self.eps, betas=self.betas, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]


class TokenClassificationModelBinary(LightningModule):
    def __init__(self, model_name: str, rubrics=None, lr=1e-5, eps=0, weight_decay=0.01, warmup_steps=0, ):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.loss = nn.BCEWithLogitsLoss()
        self.rubrics = rubrics
        # Hyperparameters
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    def create_labels(self, batch):
        targets = []
        for labels in batch:
            t = []
            for l in labels:
                if l > -100:
                    t.append(1)
                else:
                    t.append(0)
            targets.append(t)
        labels = torch.tensor(targets, dtype=torch.float32)
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
        labels = self.create_labels(labels)
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
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        data = batch
        logits, loss = self.forward(data['input_ids'], data['attention_mask'], data['labels'])
        # loss = self.loss(outputs.view(-1, outputs.size(-1)), labels)
        data['logits'] = logits
        data['loss'] = loss
        return data

    def validation_epoch_end(self, outputs):
        metric = metrics.compute_metrics(outputs)
        self.log(metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metric

    def test_step(self, batch, batch_idx):
        data = batch
        logits, loss = self.forward(data['input_ids'], data['attention_mask'], data['labels'])
        data['logits'] = logits
        data['loss'] = loss
        return data

    def test_epoch_end(self, outputs):
        metric = metrics.compute_metrics(outputs)
        self.log(metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metric

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]

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