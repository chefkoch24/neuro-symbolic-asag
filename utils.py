import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import json

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
            similarity = cosine_similarity(candidate_embedding, r)[0][0]
            similarities.append(similarity)
        return similarities