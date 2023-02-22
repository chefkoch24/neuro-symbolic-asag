import torch
import evaluate
from transformers import AutoTokenizer
import config
import numpy as np
import myutils as utils

ner_metric = evaluate.load("seqeval")
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

idx2label = {0: 'O', 1: 'I-CUE'}


def relation(y):
    # relation of how many tokens are class 1 compared to all tokens
    class_1 = 0
    class_0 = 0
    for l in y:
        if l == 1:
            class_1 += 1
        elif l == 0:
            class_0 += 1
    return class_1 / (class_0 + class_1)


def silver2target(data, th=0.5):
    targets = []
    for l in data:
        if l >= th:
            targets.append(1)
        elif l == -100:
            targets.append(-100)
        else:
            targets.append(0)
    return targets


def get_spans_from_labels(labels):
    spans = []
    prev = 0
    start_i, end_i = 0, 0
    for i, l in enumerate(labels):
        if l == 0 and prev == 1:
            end_i = i
            spans.append((start_i, end_i))
        if l == 1 and prev == 0 or l == 1 and prev == -100: #if special token
            end_i = 0
            start_i = i
        if l == -100: #if special token
            start_i = i + 1
        if i == len(labels) - 1:  # end of sequence
            if l == 1:
                end_i = i + 1  # because we want to get the last element as well
                spans.append((start_i, end_i))
        prev = l
    return spans


def partial_match(predicted_spans, true_spans):
    # Calculate the start and end indices of the predicted and true spans
    scores = []
    for predicted_span in predicted_spans:
        p_start, p_end = predicted_span[0], predicted_span[1]
        for true_span in true_spans:
            t_start, t_end = true_span[0], true_span[1]

            # Calculate the length of the overlap between the predicted and true spans
            overlap = max(0, min(p_end, t_end) - max(p_start, t_start))

            # Calculate the length of the union between the predicted and true spans
            union = max(p_end, t_end) - min(p_start, t_start)

            # Calculate the partial match score as the overlap divided by the union
            score = overlap / union if union > 0 else 0
            scores.append(score)
    # Return the average over all spans in the sequence
    return np.average(scores) if scores else 0


def get_partial_match_score(predicted_spans, true_spans):
    # Calculate the partial match score for each predicted span
    scores = [partial_match(p, t) for p in predicted_spans for t in true_spans]

    # Return the average over the batch
    return np.average(scores) if scores else 0


def get_matches(predicitons, labels):
    matches = []
    for pred, label in zip(predicitons, labels):
        if label == 0:
            matches.append(0)
        else:
            matches.append(pred / label)
    matches = sum(matches) / len(matches)
    return matches


def get_average_number_of_key_elements_by_class(labels, classes):
    num_key_elements_correct, num_key_elements_partial, num_key_elements_incorrect = [], [], []
    for label, c in zip(labels, classes):
        num_elm = len(get_spans_from_labels(label))
        if c == 'CORRECT':
            num_key_elements_correct.append(num_elm)
        elif c == 'PARTIAL_CORRECT':
            num_key_elements_partial.append(num_elm)
        elif c == 'INCORRECT':
            num_key_elements_incorrect.append(num_elm)
    return np.average(num_key_elements_correct) if num_key_elements_correct else 0.0, np.average(
        num_key_elements_partial) if num_key_elements_partial else 0.0, np.average(
        num_key_elements_incorrect) if num_key_elements_incorrect else 0.0


def get_average_realtion_by_class(labels, classes):
    relations_correct, relations_partial, relations_incorrect = [], [], []
    for label, c in zip(labels, classes):
        rel = relation(label)
        if c == 'CORRECT':
            relations_correct.append(rel)
        elif c == 'PARTIAL_CORRECT':
            relations_partial.append(rel)
        elif c == 'INCORRECT':
            relations_incorrect.append(rel)
    return np.average(relations_correct) if relations_correct else 0.0, np.average(
        relations_partial) if relations_partial else 0.0, np.average(
        relations_incorrect) if relations_incorrect else 0.0


def get_average_number_of_tokens_per_key_element_by_class(labels, classes):
    num_tokens_correct, num_tokens_partial, num_tokens_incorrect = [], [], []
    for label, c in zip(labels, classes):
        idxs = get_spans_from_labels(label)
        num_tokens = sum([end - start for start, end in idxs])
        if c == 'CORRECT':
            num_tokens_correct.append(num_tokens)
        elif c == 'PARTIAL_CORRECT':
            num_tokens_partial.append(num_tokens)
        elif c == 'INCORRECT':
            num_tokens_incorrect.append(num_tokens)
    return np.average(num_tokens_correct) if num_tokens_correct else 0.0, np.average(
        num_tokens_partial) if num_tokens_partial else 0.0, np.average(
        num_tokens_incorrect) if num_tokens_incorrect else 0.0


def compute_metrics(outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    predictions = torch.cat([x['logits'] for x in outputs]).argmax(dim=-1)  # .max(dim=-1).values
    labels = torch.cat([x['labels'] for x in outputs])
    input_ids = torch.cat([x['input_ids'] for x in outputs])
    classes = [x['class'] for x in outputs]
    classes = utils.flat_list(classes)
    # generate true values
    true_labels = [[l[1].item() for l in label if l[1] != -100] for label in labels]
    true_predictions = [
        [p.item() for (p, l) in zip(prediction, label) if l[1] != -100]
        for prediction, label in zip(predictions, labels)
    ]
    hard_labels = [silver2target(labels) for labels in true_labels]
    labels_string = [[idx2label[l] for l in label] for label in hard_labels]
    predictions_string = [[idx2label[l] for l in label] for label in true_predictions]
    # Token Metrics
    ner_metrics = ner_metric.compute(references=labels_string, predictions=predictions_string, mode='strict',
                                     scheme='IOB1')
    # Span Metrics
    true_spans = [get_spans_from_labels(l) for l in hard_labels]
    predicted_spans = [get_spans_from_labels(l) for l in true_predictions]
    pm = get_partial_match_score(predicted_spans, true_spans)
    # Justification Cue Specific Metrics
    n_correct, n_partial, n_incorrect = get_average_number_of_key_elements_by_class(true_predictions, classes)
    r_correct, r_partial, r_incorrect = get_average_realtion_by_class(true_predictions, classes)
    tn_correct, tn_partial, tn_incorrect = get_average_number_of_tokens_per_key_element_by_class(true_predictions,
                                                                                                 classes)
    metrics = {
        "seqeval_precision": ner_metrics["overall_precision"],
        "seqeval_recall": ner_metrics["overall_recall"],
        "seqeval_f1": ner_metrics["overall_f1"],
        "seqeval_accuracy": ner_metrics["overall_accuracy"],
        'val_loss': avg_loss,
        'partial_match': pm,
        'average_number_of_key_elements_correct': n_correct,
        'average_number_of_key_elements_partial': n_partial,
        'average_number_of_key_elements_incorrect': n_incorrect,
        'average_relation_correct': r_correct,
        'average_relation_partial': r_partial,
        'average_relation_incorrect': r_incorrect,
        'average_number_of_tokens_per_key_element_correct': tn_correct,
        'average_number_of_tokens_per_key_element_partial': tn_partial,
        'average_number_of_tokens_per_key_element_incorrect': tn_incorrect,
    }

    return metrics
