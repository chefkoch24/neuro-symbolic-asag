import torch
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
from evaluate import load


class ParaphraseScorerSBERT():
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

    def detect_score_key_elements(self, candidate, rubric):
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

class BertScorer():
    def __init__(self, model_name='xlm-roberta-large'):
        self.bertscore = load("bertscore")
        self.model_name = model_name

    def detect_score_key_elements(self, key_element, rubric):
        references = [key_element] * len(rubric['key_element'])
        predictions = rubric['key_element'].tolist()
        if len(key_element) == 0:
            return [0] * len(rubric['key_element'])
        try:
            return self. bertscore.compute(predictions=predictions, references=references,
                          model_type=self.model_name)['f1']
        except:
            return [0] * len(rubric['key_element'])