import torch
import evaluate
from torchmetrics import F1Score, Accuracy, Precision, Recall, MeanSquaredError, CohenKappa
import numpy as np
import myutils as utils
import statistics


idx2label = {0: 'O', 1: 'I-CUE'}
span_metric = evaluate.load("squad")
rubrics = utils.load_json('data/rubrics.json')

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
    scores = [partial_match(p, t) for p,t in  zip(predicted_spans, true_spans)]

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

def micro_macro_f1(predictions, labels):
    # Flatten the lists of predictions and labels
    y_true = [label for label_list in labels for label in label_list]
    y_pred = [pred for pred_list in predictions for pred in pred_list]

    # Calculate true positives, false positives, and false negatives for each label
    tp = [0] * 2
    fp = [0] * 2
    fn = [0] * 2
    for i in range(0,len(y_pred)):
        if y_true[i] == y_pred[i]:
            tp[y_true[i]] += 1
        else:
            fp[y_pred[i]] += 1
            fn[y_true[i]] += 1

    # Calculate micro-averaged F1 score, precision, and recall
    micro_tp = sum(tp)
    micro_fp = sum(fp)
    micro_fn = sum(fn)
    if micro_tp == 0:
        micro_precision = 0
        micro_recall = 0
        micro_f1_score = 0
    else:
        micro_precision = micro_tp / (micro_tp + micro_fp)
        micro_recall = micro_tp / (micro_tp + micro_fn)
        micro_f1_score = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall))

    # Calculate macro-averaged F1 score, precision, and recall
    macro_precision = sum(tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] != 0 else 0 for i in range(2)) / 2
    macro_recall = sum(tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] != 0 else 0 for i in range(2)) / 2
    if macro_precision == 0 and macro_recall == 0:
        macro_f1_score = 0
    else:
        macro_f1_score = 2 * ((macro_precision * macro_recall) / (macro_precision + macro_recall))

    # Calculate accuracy
    accuracy = sum(1 for i in range(len(y_pred)) if y_true[i] == y_pred[i]) / len(y_pred)

    return {
        'micro_f1': micro_f1_score,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_f1': macro_f1_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'accuracy': accuracy,
    }
def get_statistical_metrics(labels, classes):
    corrects, partial_corrects, incorrects = [],[],[]
    for label, c in zip(labels, classes):
        if c == 'CORRECT':
            corrects.append(label)
        elif c == 'PARTIAL_CORRECT':
            partial_corrects.append(label)
        elif c == 'INCORRECT':
            incorrects.append(label)
    corrects = utils.flat_list(corrects)
    partial_corrects = utils.flat_list(partial_corrects)
    incorrects = utils.flat_list(incorrects)
    metrics ={
        'average_correct': np.average(corrects) if corrects else 0.0,
        'average_partial': np.average(partial_corrects) if partial_corrects else 0.0,
        'average_incorrect': np.average(incorrects) if incorrects else 0.0,
        'std_correct': np.std(corrects) if corrects else 0.0,
        'std_partial': np.std(partial_corrects) if partial_corrects else 0.0,
        'std_incorrect': np.std(incorrects) if incorrects else 0.0,
        'median_correct': np.median(corrects) if corrects else 0.0,
        'median_partial': np.median(partial_corrects) if partial_corrects else 0.0,
        'median_incorrect': np.median(incorrects) if incorrects else 0.0,
        'mode_correct': statistics.mode(corrects) if corrects else 0.0,
        'mode_partial': statistics.mode(partial_corrects) if partial_corrects else 0.0,
        'mode_incorrect': statistics.mode(incorrects) if incorrects else 0.0,
        'min_correct': min(corrects) if corrects else 0.0,
        'min_partial': min(partial_corrects) if partial_corrects else 0.0,
        'min_incorrect': min(incorrects) if incorrects else 0.0,
        'max_correct': max(corrects) if corrects else 0.0,
        'max_partial': max(partial_corrects) if partial_corrects else 0.0,
        'max_incorrect': max(incorrects) if incorrects else 0.0,
        'labeled_tokens_correct': len([c for c in corrects if c > 0]),
        'labeled_tokens_partial': len([c for c in partial_corrects if c > 0]),
        'labeled_tokens_incorrect': len([c for c in incorrects if c > 0]),
    }
    return metrics


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
        spans = get_spans_from_labels(label)
        for start, end in spans:
            num_tokens = end - start
            if c == 'CORRECT':
                num_tokens_correct.append(num_tokens)
            elif c == 'PARTIAL_CORRECT':
                num_tokens_partial.append(num_tokens)
            elif c == 'INCORRECT':
                num_tokens_incorrect.append(num_tokens)
    return np.average(num_tokens_correct) if num_tokens_correct else 0.0, np.average(
        num_tokens_partial) if num_tokens_partial else 0.0, np.average(
        num_tokens_incorrect) if num_tokens_incorrect else 0.0


def compute_metrics_token_classification(outputs):
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
    # Token Metrics
    ner_metrics = micro_macro_f1(predictions=true_predictions, labels=hard_labels)
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
        "macro_precision": ner_metrics["macro_precision"],
        "macro_recall": ner_metrics["macro_recall"],
        "macro_f1": ner_metrics["macro_f1"],
        "micro_precision": ner_metrics["micro_precision"],
        "micro_recall": ner_metrics["micro_recall"],
        "micro_f1": ner_metrics["micro_f1"],
        "accuracy": ner_metrics["accuracy"],
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

def compute_f1_spans(pred_span, true_span):
    pred_tokens = set(range(pred_span[0][0], pred_span[0][1] + 1))
    true_tokens = set(range(true_span[0][0], true_span[0][1] + 1))
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return 0, 0, 0
    precision = len(pred_tokens & true_tokens) / len(pred_tokens)
    recall = len(pred_tokens & true_tokens) / len(true_tokens)
    if precision == 0 or recall == 0:
        return 0, 0, 0  # all values are 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def compute_metrics_span_prediction(outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    start_positions= torch.cat([x['start_positions'] for x in outputs])
    end_positions = torch.cat([x['end_positions'] for x in outputs])
    attention_mask = torch.cat([x['attention_mask'] for x in outputs])
    token_type_ids = torch.cat([x['token_type_ids'] for x in outputs])
    # mask the start and end predictions with attention mask and token type ids
    start_logits = torch.cat([x['start_logits'] for x in outputs])
    end_logits = torch.cat([x['end_logits'] for x in outputs])
    mask = (attention_mask == 1) & (token_type_ids == 1)
    start_logits_masked = start_logits.masked_fill(~mask, float('-inf'))
    end_logits_masked = end_logits.masked_fill(~mask, float('-inf'))

    start_predictions = start_logits_masked.argmax(dim=-1)
    end_predictions = end_logits_masked.argmax(dim=-1)

    input_ids = torch.cat([x['input_ids'] for x in outputs])
    classes = [x['class'] for x in outputs]
    classes = utils.flat_list(classes)

    # for using the same method it's wrapped in []
    predicted_spans = [[(s.item(), e.item())] for s, e in zip(start_predictions, end_predictions)]
    true_spans = [[(s.item(), e.item())] for s, e in zip(start_positions, end_positions)]
    pm = get_partial_match_score(true_spans, predicted_spans)
    f1s_precisions_recalls = [compute_f1_spans(p_span, t_span) for p_span, t_span in zip(predicted_spans, true_spans)]
    f1 = np.average([v[0] for v in f1s_precisions_recalls])
    precision = np.average([v[1] for v in f1s_precisions_recalls])
    recall = np.average([v[2] for v in f1s_precisions_recalls])
    return {
        'val_loss': avg_loss,
        'partial_match': pm,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def compute_grading_classification_metrics(outputs):
    predictions = torch.cat([x['prediction'] for x in outputs])
    labels = torch.cat([x['class'] for x in outputs])
    langs = utils.flat_list([x['lang'] for x in outputs])
    f1_score = F1Score(task='multiclass', num_classes=3, average='none')
    accuracy = Accuracy(task='multiclass', num_classes=3, average='none')
    precision = Precision(task='multiclass', num_classes=3, average='none')
    recall = Recall(task='multiclass', num_classes=3, average='none')
    f1 = f1_score(predictions, labels)
    accu = accuracy(predictions, labels)
    prec = precision(predictions, labels)
    rec = recall(predictions, labels)
    macro_f1_score = F1Score(task='multiclass', num_classes=3, average='macro')
    weighted_f1_score = F1Score(task='multiclass', num_classes=3, average='weighted')
    accuracy = Accuracy(task='multiclass', num_classes=3)
    mf1s, wf1s, accs = [], [], []
    predictions = predictions.argmax(-1).detach().numpy().tolist()
    labels = labels.detach().numpy().tolist()
    for language in ['de', 'en']:
        tmp_predictions = [p for p,l in zip(predictions, langs) if l == language]
        tmp_labels = [p for p,l in zip(labels, langs) if l == language]
        tmp_labels = torch.tensor(tmp_labels)
        tmp_predictions = torch.tensor(tmp_predictions)
        mf1 = macro_f1_score(tmp_predictions, tmp_labels)
        wf1 = weighted_f1_score(tmp_predictions, tmp_labels)
        acc = accuracy(tmp_predictions, tmp_labels)
        mf1s.append(mf1)
        wf1s.append(wf1)
        accs.append(acc)
    return {
        'val_f1_correct': f1[0].item(),
        'val_f1_partial': f1[1].item(),
        'val_f1_incorrect': f1[2].item(),
        'val_acc_correct': accu[0].item(),
        'val_acc_partial': accu[1].item(),
        'val_acc_incorrect': accu[2].item(),
        'val_precision_correct': prec[0].item(),
        'val_precision_partial': prec[1].item(),
        'val_precision_incorrect': prec[2].item(),
        'val_recall_correct': rec[0].item(),
        'val_recall_partial': rec[1].item(),
        'val_recall_incorrect': rec[2].item(),
        'macro_f1_de': mf1s[0],
        'macro_f1_en': mf1s[1],
        'weighted_f1_de': wf1s[0],
        'weighted_f1_en': wf1s[1],
        'accuracy_de': accs[0],
        'accuracy_en': accs[1],
        'val_loss': torch.stack([x['loss'] for x in outputs]).mean(),
    }

def compute_grading_regression_metrics(outputs):
    predictions = torch.cat([x['prediction'] for x in outputs]).numpy().tolist()
    targets = torch.cat([x['score'] for x in outputs]).numpy().tolist()
    langs = utils.flat_list([x['lang'] for x in outputs])
    rmse_calc = MeanSquaredError(squared=False)
    cohenkappa = CohenKappa(task="multiclass", num_classes=2, weights="quadratic")
    rmses, qwks = [], []
    for language in ['de', 'en']:
        tmp_predictions = [p for p,l in zip(predictions, langs) if l == language]
        tmp_labels = [p for p,l in zip(targets, langs) if l == language]
        tmp_labels = torch.tensor(tmp_labels)
        tmp_predictions = torch.tensor(tmp_predictions)
        rmse = rmse_calc(tmp_predictions, tmp_labels)
        qwk = cohenkappa(tmp_predictions, tmp_labels)
        rmses.append(rmse.item())
        qwks.append(qwk.item())

    return {
        'rmse_de': rmses[0],
        'rmse_en': rmses[1],
        'qwk_de': qwks[0],
        'qwk_en': qwks[1],
        'val_loss': torch.stack([x['loss'] for x in outputs]).mean(),
    }