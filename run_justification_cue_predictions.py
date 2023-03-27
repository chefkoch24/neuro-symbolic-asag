from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from config import Config
import myutils as utils
from dataset import *
from paraphrase_scorer import BertScorer
from preprocessor import *
from justification_cue_model import *

config = Config(
   # task='token_classification',
    task='span_prediction',
    model='distilbert-base-multilingual-cased',
    dev_file='dev_aligned_labels_distilbert-base-multilingual-cased.json',
    #checkpoint_path='logs/token_classification_distilbert-base-multilingual-cased_None/version_0/checkpoints/checkpoint-epoch=02-val_loss=0.64.ckpt',
    checkpoint_path='logs/span_prediction_distilbert-base-multilingual-cased/version_0/checkpoints/checkpoint-epoch=01-val_loss=3.56.ckpt',
    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    context=True,
)

experiment_name = utils.get_experiment_name(['predictions', config.TASK, config.MODEL_NAME, config.CONTEXT])

def post_process_token_classification(predictions, dataset):
    predictions = torch.cat([x for x in predictions])
    results = []
    for p,d in zip(predictions, dataset):
       p = p.cpu().numpy()
       attention_mask = d['attention_mask'].cpu().numpy() == 1
       token_type_ids = d['token_type_ids'].cpu().numpy() == 0
       valid_indices = attention_mask & token_type_ids
       p = p[valid_indices]
       input_ids = d['input_ids'].cpu().numpy()[valid_indices]
       # remove CLS and SEP
       input_ids = input_ids[1:-1]
       p = p[1:-1]
       pred_spans = metrics.get_spans_from_labels(p)
       pred_spans = [tokenizer.decode(input_ids[s[0]:s[1]]) for s in pred_spans]
       # true spans
       true_labels = [torch.argmax(l, axis=-1).item() for l in d['labels'] if l[1] != -100]
       spans = metrics.get_spans_from_labels(true_labels)
       true_spans = [tokenizer.decode(input_ids[s[0]:s[1]]) for s in spans]
       results.append({
           'question_id': d['question_id'],
           'student_answer': d['student_answer'],
           'true_spans': true_spans,
           'pred_spans': pred_spans,
           'class': d['class']
       })
    return results

def post_process_span_prediction(predictions, dataset):
    results = []
    starts, ends = zip(*predictions)
    starts = utils.flat_list(np.array([x.cpu().numpy() for x in starts]).tolist())
    ends = utils.flat_list(np.array([x.cpu().numpy() for x in ends]).tolist())
    for start,end, d in zip(starts, ends, dataset):
        mask = (d['token_type_ids'] == 1) & (d['attention_mask'] == 1)
        input_ids = d['input_ids'].masked_fill(~mask, int(-1))
        span = ''
        if start < end:
            span = input_ids[start:end]
            span = tokenizer.decode(span)
        results.append({
                'question_id': d['question_id'],
                'student_answer': d['student_answer'],
                'prediction': span,
                'rubric_element': d['rubric_element'],
                'class': d['class']
            })
    return results


trainer = Trainer()
dev_data = utils.load_json(config.PATH_DATA + '/' + config.DEV_FILE)[0:10]
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
scorer = BertScorer()
if config.TASK == 'span_prediction':
    preprocessor = PreprocessorSpanPrediction(config.MODEL_NAME, scorer=scorer, rubrics=rubrics)
    post_proccess_function = post_process_span_prediction
    model = SpanPredictionModel.load_from_checkpoint(checkpoint_path=config.PATH_CHECKPOINT,
                                                     map_location=config.DEVICE)
    Dataset = SpanJustificationCueDataset
elif config.TASK == 'token_classification':
    preprocessor = PreprocessorTokenClassification(config.MODEL_NAME, with_context=config.CONTEXT)
    post_proccess_function = post_process_token_classification
    model = TokenClassificationModel.load_from_checkpoint(checkpoint_path=config.PATH_CHECKPOINT,
                                                          map_location=config.DEVICE)
    Dataset = JustificationCueDataset

dev_dataset = preprocessor.preprocess(dev_data)
dev_dataset = Dataset(dev_dataset)
val_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

predictions = trainer.predict(model, val_loader, return_predictions=True)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

results = post_proccess_function(predictions, dev_dataset)
utils.save_csv(results, config.PATH_RESULTS, experiment_name, with_timestamp=True)

