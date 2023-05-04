import spacy
from nltk import PorterStemmer
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_lg")
stemmer = PorterStemmer()

sentence = 'My lazy dog jumped over the fence of our neighbor\'s yard.'
doc = nlp(sentence)

stemmed = [stemmer.stem(token.text) for token in doc]
lemmatized = [token.lemma_ for token in doc]

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
input_ids = tokenizer.encode(sentence, add_special_tokens=False)
sentence = [tokenizer.decode([input_id]) for input_id in input_ids]

print('Word-piece tokens:', sentence)
print('Stemmed:', stemmed)
print('Lemmatized:', lemmatized)
