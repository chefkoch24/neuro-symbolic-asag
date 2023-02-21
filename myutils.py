import os

import pandas as pd
import skweak
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import torch
import json

import config


def save_json(data, path, file_name):
    directory = path
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(path+'/'+file_name, 'w') as fout:
        json.dump(data, fout)
        print('saved', file_name)

def load_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data

def flat_list(lst):
    x = [item for sublist in lst for item in sublist]
    return x

def save_to_csv(X_train, X_dev, y_train, y_dev, path):
    sep = "\t"
    directory = path

    if not os.path.exists(directory):
        os.mkdir(directory)

    save_path = path +'/'
    X_train.to_csv(save_path +"x_train.tsv", sep=sep)
    X_dev.to_csv(save_path+"x_dev.tsv", sep=sep)
    pd.DataFrame(data=y_train).to_csv(save_path+"y_train.tsv", sep=sep)
    pd.DataFrame(data=y_dev).to_csv(save_path+ "y_dev.tsv", sep=sep)
    print('successfully saved')

def load_rubrics(path):
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    rubrics = dict()
    for key in data:
        rubrics[key] = pd.DataFrame.from_dict(data[key])
    return rubrics

def prepare_rubrics(rubrics):
    german_question_ids = [str(i) for i in range(1, 10)]
    for key in rubrics:
        rubric = rubrics[key]
        tokenezied_elements = []
        for i, r in rubric.iterrows():
            key_element = r['key_element']
            if key in german_question_ids:
                tokenized = config.nlp_de(key_element)
            else:
                tokenized = config.nlp(key_element)
            tokenezied_elements.append(tokenized)
        rubric['tokenized'] = tokenezied_elements
        rubrics[key] = rubric
    return rubrics

def save_annotated_corpus(annotated_docs, path):
    print(len(annotated_docs))
    for doc in annotated_docs:
        doc.ents = doc.spans["hmm"]
    skweak.utils.docbin_writer(annotated_docs, path)

def tokenize_data(data):
    tokenized = []
    for _, d in data.iterrows():
        if d['lang'] == 'en':
            d = config.nlp(d['student_answer'])
        elif d['lang'] == 'de':
            d = config.nlp_de(d['student_answer'])
        tokenized.append(d)
    data['tokenized'] = tokenized
    return data

def create_labels_probability_distribution(labels):
    targets = []
    for l in labels:
        if l > -100:
            targets.append([1 - l, l])
        else:
            targets.append([l, l])
    return targets


class ParaphraseDetector():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

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
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def detect_paraphrases(self, candidate, rubric):
        rubric_elements = self._encode_rubric(rubric)
        # encode the candidate
        encoded_input = self.tokenizer(candidate, is_split_into_words=True, padding=True, truncation=True,
                                       return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            candidate_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        similarities = []
        for r in rubric_elements:
            similarity = cosine(candidate_embedding[0], r[0])
            similarities.append(similarity)
        return similarities
