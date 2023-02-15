import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AdamW
from torch.optim import lr_scheduler
import logging
import metrics

class GradingModelScore(LightningModule):
    def __init__(self, model_name: str, rubrics=None):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, output_hidden_states=True)
        self.loss = nn.CrossEntropyLoss()
        self.rubrics = rubrics

    def _encode_rubric(self, rubric):
        sentence_embeddings = []
        for r in rubric['key_element']:
            encoded_input = self.tokenizer(r, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings.append(sentence_embedding)
        return sentence_embeddings

    # Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def score_similarities(self, candidate, rubric):
        rubric_elements = self._encode_rubric(rubric)
        # encode the candidate
        encoded_input = self.tokenizer(candidate, is_split_into_words=True, padding=True, truncation=True,
                                       return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            candidate_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        similarities = []
        for r in rubric_elements:
            similarity = cosine_similarity(candidate_embedding, r)[0][0]
            similarities.append(similarity)
        return similarities

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