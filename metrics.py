import torch
import evaluate
from transformers import AutoTokenizer
import config


ner_metric = evaluate.load("seqeval")
exact_match_metric = evaluate.load("evaluate-metric/exact_match")
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

idx2label = {0: 'O', 1: 'I-CUE'}

def relation(y):
    # relation of how many tokens are class 1 compared to all tokens
    class_1 = 0
    class_0 = 0
    for l in y:
            if l == 1:
                class_1 +=1
            elif l == 0:
                class_0 +=1
    return class_1/(class_0+class_1)

def silver2target(data, th=0.5):
    targets = []
    for l in data:
        if l >= th:
            targets.append(1)
        else:
            targets.append(0)
    return targets


def get_idxs_elements(labels):
    indicies = []
    prev = 0
    start_i, end_i = 0, 0
    for i, l in enumerate(labels):
        if l == 0 and prev == 1:
            end_i = i
            indicies.append((start_i, end_i))
        if l == 1 and prev == 0:
            end_i = 0
            start_i = i
        if i == len(labels) - 1:  # end of sequence
            if l == 1:
                end_i = i + 1  # because we want to get the last element as well
                indicies.append((start_i, end_i))
        prev = l
    return indicies

def compute_metrics(outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    predictions = torch.cat([x['logits'] for x in outputs]).argmax(dim=-1)  # .max(dim=-1).values
    labels = torch.cat([x['labels'] for x in outputs])
    input_ids = torch.cat([x['input_ids'] for x in outputs])
    # generate true values
    true_labels = [[l.item() for l in label if l != -100] for label in labels]
    true_predictions = [
        [p.item() for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    hard_labels = [silver2target(labels) for labels in true_labels]
    labels_string = [[idx2label[l] for l in label] for label in hard_labels]
    predictions_string = [[idx2label[l] for l in label] for label in true_predictions]
    # Token Metrics
    ner_metrics = ner_metric.compute(references=labels_string, predictions=predictions_string, mode='strict', scheme='IOB1')
    idx_labels = [get_idxs_elements(labels) for labels in hard_labels]
    idx_predictions = [get_idxs_elements(labels) for labels in true_predictions]

    # Matches of found rubrics in the answer
    idx_predictions = [len(pred) for pred in idx_predictions]
    idx_labels = [len(label) for label in idx_labels]
    # Calculate the number of found key elements in the answer with respect to the number of annotated key elements in the answer
    matches = []
    for pred, label in zip(idx_predictions, idx_labels):
        if label == 0:
            matches.append(0)
        else:
            matches.append(pred/label)
    matches = sum(matches)/len(matches)

    relations = [relation(pred) for pred in true_predictions]
    relations = sum(relations)/len(relations)

    metrics = {
        "seqeval_precision": ner_metrics["overall_precision"],  # exact match for spans is the same as precision on token level
        "seqeval_recall": ner_metrics["overall_recall"],
        "seqeval_f1": ner_metrics["overall_f1"],
        "seqeval_accuracy": ner_metrics["overall_accuracy"],
        'val_loss': avg_loss,
        'matches': matches,
        'relation': relations,
    }

    return metrics