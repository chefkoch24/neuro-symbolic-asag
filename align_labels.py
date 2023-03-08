from tqdm import tqdm
from transformers import AutoTokenizer

import config
import myutils as utils
import utils_preprocessing


def create_aligned_labels(data):
    aligned_data = []
    for d in tqdm(data):
        student_answer = d['student_answer']
        labels = d['silver_labels']
        nlp = config.nlp_de if d['lang'] == 'de' else config.nlp
        tokens_spacy = [t.text for t in nlp(student_answer)]
        # Tokenize the input to generate alignment
        tokenized = tokenizer(student_answer, add_special_tokens=False, return_offsets_mapping=True)
        tokens_bert = [tokenizer.decode(t) for t in tokenized['input_ids']] # used for alingment
        aligned_labels = utils_preprocessing.align_generate_labels_all_tokens(tokens_spacy, tokens_bert, labels).tolist()
        # get the spans from the aligned labels
        d['aligned_labels'] = aligned_labels
        aligned_data.append(d)
    return aligned_data


train_data = utils.load_json(config.PATH_DATA + '/' + config.ANNOTATED_TRAIN_FILE)
dev_data = utils.load_json(config.PATH_DATA + '/' + config.ANNOTATED_DEV_FILE)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
rubrics = utils.load_rubrics(config.PATH_RUBRIC)

training_dataset = create_aligned_labels(train_data)
dev_dataset = create_aligned_labels(dev_data)

#save data
DATASET_NAME = 'dataset_aligned_labels_' + config.MODEL_NAME + '.json'
utils.save_json(training_dataset, config.PATH_DATA + '/', 'training_' + DATASET_NAME)
utils.save_json(dev_dataset, config.PATH_DATA + '/', 'dev_'+DATASET_NAME)