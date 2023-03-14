# This script annotates the student answers with the respective labeling functions

# IMPORTS
import sklearn
import pandas as pd
import re
import spacy
import numpy as np
import nltk

import myutils
from paraphrase_scorer import BertScorer

from nltk.stem.porter import *
from rouge import Rouge
from nltk.metrics.distance import *
from spacy.matcher import PhraseMatcher
import warnings
from tqdm import tqdm
import config
import myutils

# Filter out warnings from the NLTK package
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")


class WeakSupervisionSoft():
    def __init__(self, rubrics=None):
        self.rubrics = rubrics
        self.para_detector = BertScorer()
        self._punctuation = ['.', ',', '?', '!', ';', ':']
        self.rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-3", "rouge-4", "rouge-5", "rouge-l"])
        self.labeling_functions = [
            {'name': 'LF_noun_phrases', 'function': self.LF_noun_phrases},
            {'name': 'LF_lemma_match', 'function': self.LF_lemma_match},
            {'name': 'LF_pos_match', 'function': self.LF_pos_match},
            {'name': 'LF_lemma_match_without_stopwords', 'function': self.LF_lemma_match_without_stopwords},
            {'name': 'LF_pos_match_without_stopwords', 'function': self.LF_pos_match_without_stopwords},
            {'name': 'LF_shape_match', 'function': self.LF_shape_match},
            {'name': 'LF_stem_match', 'function': self.LF_stem_match},
            {'name': 'LF_tag_match', 'function': self.LF_tag_match},
            {'name': 'LF_dep_match', 'function': self.LF_dep_match},
            {'name': 'LF_dep_match_without_stopwords', 'function': self.LF_dep_match_without_stopwords},
            {'name': 'LF_bi_gram_overlap', 'function': self.LF_bi_gram_overlap},
            {'name': 'LF_tri_gram_overlap', 'function': self.LF_tri_gram_overlap},
            {'name': 'LF_tetra_gram_overlap', 'function': self.LF_tetra_gram_overlap},
            {'name': 'LF_penta_gram_overlap', 'function': self.LF_penta_gram_overlap},
            {'name': 'LF_rouge_1_candidate', 'function': self.LF_rouge_1_candidate},
            {'name': 'LF_rouge_2_candidate', 'function': self.LF_rouge_2_candidate},
            {'name': 'LF_rouge_L_candidate', 'function': self.LF_rouge_L_candidate},
            {'name': 'LF_rouge_L_sentences', 'function': self.LF_rouge_L_sentences},
            {'name': 'LF_word_alignment', 'function': self.LF_word_alignment},
            {'name': 'LF_edit_distance', 'function': self.LF_edit_distance},
            {'name': 'LF_paraphrase_detection_sentences', 'function': self.LF_paraphrase_detection_sentences},
            {'name': 'LF_paraphrase_detection_candidates', 'function': self.LF_paraphrase_detection_candidates},
            {'name': 'LF_bleu_candidates', 'function': self.LF_bleu_candidates},
            {'name': 'LF_meteor_candidates', 'function': self.LF_meteor_candidates},
            {'name': 'LF_meteor_sentences', 'function': self.LF_meteor_sentences},
            {'name': 'LF_jaccard_similarity', 'function': self.LF_jaccard_similarity},
        ]

    def _get_rubric(self, question_id: str):
        return self.rubrics[question_id]

    def _generate_sentences(self, tokenized_sequence):
        sentences = []
        indicies = []
        for sent in tokenized_sequence.sents:
            if sent[-1].text in self._punctuation:
                sent = sent[:-1]
            sentences.append(sent)
            indicies.append((sent.start, sent.end))
        return sentences, indicies

    def _generate_candidates(self, tokenized_sequence):
        candidate = []
        candidates, indicies = [], []
        is_first_sentence = True
        for i, t in enumerate(tokenized_sequence):
            if t.text in self._punctuation:
                if len(candidate) > 0:
                    candidates.append(candidate)
                    candidate = []
            elif t.pos_ != 'SPACE':
                candidate.append(t)
                # append also the last elements if no punctuation symbol was used
        if len(candidate) > 0:
            candidates.append(candidate)
        indicies = [(c[0].i, c[-1].i + 1) for c in candidates]  # +1 to get the last token
        return candidates, indicies

    def LF_paraphrase_detection_candidates(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        for c, i in zip(candidates, indicies):
            c = [t.text for t in c]
            sim = self.para_detector.detect_score_key_elements(c, rubric)
            max_index = np.argmax(sim)
            max_value = sim[max_index]
            yield i[0], i[-1], 'CUE', max_value

    def LF_paraphrase_detection_sentences(self, doc, rubric, lang):
        candidates, indicies = self._generate_sentences(doc)
        for c, i in zip(candidates, indicies):
            c = [t.text for t in c]
            sim = self.para_detector.detect_score_key_elements(c, rubric)
            max_index = np.argmax(sim)
            max_value = sim[max_index]
            yield i[0], i[-1], 'CUE', max_value

    def LF_ner(self, doc, rubric, lang):
        ind_dict = dict((k, i) for i, k in enumerate(doc))
        for ents in doc.ents:
            matches = set(doc).intersection(ents)
            indices = [ind_dict[x] for x in matches]
            indices = sorted(indices)
            if indices != []:
                yield indices[0], indices[-1], 'CUE'

    def LF_noun_phrases(self, doc, rubric, lang):
        noun_chunks = [chunk for chunk in doc.noun_chunks]
        noun_chunk_text = []
        for nc in noun_chunks:
            noun_chunk_text.append(' '.join([t.text.lower() for t in nc]))
        ind_dict = dict((k, i) for i, k in enumerate(noun_chunk_text))
        rubric_text = []
        for r in rubric['key_element']:
            rubric_text.append(r.lower())
        matches = set(noun_chunk_text).intersection(rubric_text)
        indices = [ind_dict[x] for x in matches]
        for i in indices:
            yield noun_chunks[i].start, noun_chunks[i].end, 'CUE', 1.0

    def LF_meteor_candidates(self, doc, rubric, lang):
        references = []
        for r in rubric['tokenized']:
            key_element = r
            reference = [t.text.lower() for t in key_element]
            references.append(reference)
        labels = []
        candidates, indicies = self._generate_candidates(doc)
        # meteor expects list of tokens
        for c in candidates:
            scores = []
            hypothesis = [t.text.lower() for t in c]
            for r in references:
                score = nltk.translate.meteor_score.single_meteor_score(hypothesis, r)
                scores.append(score)
            labels.append(scores)
        labels = np.array(labels)
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def LF_meteor_sentences(self, doc, rubric, lang):
        references = []
        for r in rubric['tokenized']:
            key_element = r
            reference = [t.text.lower() for t in key_element]
            references.append(reference)
        labels = []
        candidates, indicies = self._generate_sentences(doc)
        for c in candidates:
            scores = []
            hypothesis = [t.text.lower() for t in c]
            for r in references:
                score = nltk.translate.meteor_score.single_meteor_score(hypothesis, r)
                scores.append(score)
            labels.append(scores)
        labels = np.array(labels)
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def LF_stem_match(self, doc, rubric, lang):
        # include stop words
        stemmer = PorterStemmer()
        doc = [stemmer.stem(token.text.lower()) for token in doc]
        stemmed_rubric = []
        for r in rubric['tokenized']:
            stemmed_rubric = [stemmer.stem(token.text.lower()) for token in r]
            ind_dict = dict((k, i) for i, k in enumerate(doc))
            matches = set(doc).intersection(stemmed_rubric)
            indices = [ind_dict[x] for x in matches]
            indices = sorted(indices)
            if indices != []:
                if indices[0] != indices[-1]:
                    yield indices[0], indices[-1], 'CUE', 1.0

    def LF_pos_match_without_stopwords(self, doc, rubric, lang):
        # remove stop words
        doc = [token.pos_.lower() for token in self._remove_stopwords(doc, lang)]
        stemmed_rubric = []
        for r in rubric['tokenized']:
            stemmed_rubric = [token.pos_.lower() for token in self._remove_stopwords(r, lang)]
            ind_dict = dict((k, i) for i, k in enumerate(doc))
            matches = set(doc).intersection(stemmed_rubric)
            indices = [ind_dict[x] for x in matches]
            indices = sorted(indices)
            if indices != []:
                if indices[0] != indices[-1]:
                    yield indices[0], indices[-1], 'CUE', 1.0

    def LF_lemma_match_without_stopwords(self, doc, rubric, lang):
        # remove stop words
        doc = [token.lemma_.lower() for token in self._remove_stopwords(doc, lang)]
        stemmed_rubric = []
        for r in rubric['tokenized']:
            stemmed_rubric = [token.lemma_.lower() for token in self._remove_stopwords(r, lang)]
            ind_dict = dict((k, i) for i, k in enumerate(doc))
            matches = set(doc).intersection(stemmed_rubric)
            indices = [ind_dict[x] for x in matches]
            indices = sorted(indices)
            if indices != []:
                if indices[0] != indices[-1]:
                    yield indices[0], indices[-1], 'CUE', 1.0

    def LF_dep_match_without_stopwords(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        dep_rubric = []
        for r in rubric['tokenized']:
            dep_rubric.append([t.dep_ for t in self._remove_stopwords(r, lang)])
        for c, i in zip(candidates, indicies):
            c = [t.dep_ for t in self._remove_stopwords(c, lang)]
            if c in dep_rubric:
                yield i[0], i[-1], 'CUE', 1.0

    def LF_lemma_match(self, doc, rubric, lang):
        nlp = config.nlp_de if lang == 'de' else config.nlp
        matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
        patterns = rubric['tokenized'].tolist()
        matcher.add("Lemma_Match", patterns)
        sentences, indices = self._generate_sentences(doc)
        for s in sentences:
            for match_id, start, end in matcher(s):
                yield start, end, "CUE", 1.0

    def LF_pos_match(self, doc, rubric, lang):
        nlp = config.nlp_de if lang == 'de' else config.nlp
        matcher = PhraseMatcher(nlp.vocab, attr="POS")
        patterns = rubric['tokenized'].tolist()
        matcher.add("POS_Match", patterns)
        sentences, indices = self._generate_sentences(doc)
        for s in sentences:
            for match_id, start, end in matcher(s):
                yield start, end, "CUE", 1.0

    def LF_dep_match(self, doc, rubric, lang):
        nlp = config.nlp_de if lang == 'de' else config.nlp
        matcher = PhraseMatcher(nlp.vocab, attr="DEP")
        patterns = rubric['tokenized'].tolist()
        matcher.add("DEP_Match", patterns)
        sentences, indices = self._generate_sentences(doc)
        for s in sentences:
            for match_id, start, end in matcher(s):
                yield start, end, "CUE", 1.0

    def LF_shape_match(self, doc, rubric, lang):
        nlp = config.nlp_de if lang == 'de' else config.nlp
        matcher = PhraseMatcher(nlp.vocab, attr="SHAPE")
        patterns = rubric['tokenized'].tolist()
        matcher.add("Shape_Match", patterns)
        sentences, indices = self._generate_sentences(doc)
        for s in sentences:
            for match_id, start, end in matcher(s):
                yield start, end, "CUE", 1.0

    def LF_tag_match(self, doc, rubric, lang):
        nlp = config.nlp_de if lang == 'de' else config.nlp
        matcher = PhraseMatcher(nlp.vocab, attr="TAG")
        patterns = rubric['tokenized'].tolist()
        matcher.add("Tag_Match", patterns)
        sentences, indices = self._generate_sentences(doc)
        for s in sentences:
            for match_id, start, end in matcher(s):
                yield start, end, "CUE", 1.0

    def _remove_stopwords(self, tokenized_sequence, lang):
        nlp = config.nlp_de if lang == 'de' else config.nlp
        # TODO: custom list, remove negations from stop words
        stop_words = nlp.Defaults.stop_words
        return [t for t in tokenized_sequence if t.text not in stop_words]

    def LF_bleu_candidates(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        for c, i in zip(candidates, indicies):
            c = ' '.join([t.text for t in c])
            scores = []
            for r in rubric['key_element']:
                score = nltk.translate.bleu_score.sentence_bleu([r.lower()], c.lower(), weights=(1, 0, 0, 0))
                scores.append(score)
            v = np.argmax(scores)
            if scores[v] > 0.0:
                yield i[0], i[1], 'CUE', scores[v]

    def LF_word_alignment(self, doc, rubric, lang):
        references = []
        for r in rubric['tokenized']:
            key_element = r
            reference = [t.text.lower() for t in key_element]
            references.append(reference)
        hypothesis = [t.text.lower() for t in doc]
        for r in references:
            alignment = nltk.translate.meteor_score.align_words(hypothesis, r)
            a_sorted = sorted(alignment[0], key=lambda tup: tup[0])
            for a in a_sorted:
                yield a[0], a[0] + 1, "CUE", 1.0

    def LF_bi_gram_overlap(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        labels = self._n_gram_lemma_overlap(candidates, rubric, n_gram=2)
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def LF_tri_gram_overlap(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        labels = self._n_gram_lemma_overlap(candidates, rubric, n_gram=3)
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def LF_tetra_gram_overlap(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        labels = self._n_gram_lemma_overlap(candidates, rubric, n_gram=4)
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def LF_penta_gram_overlap(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        labels = self._n_gram_lemma_overlap(candidates, rubric, n_gram=5)
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def _n_gram_lemma_overlap(self, candidates, rubric, n_gram=2):
        labels = []
        for c in candidates:
            c = ' '.join([t.lemma_.lower() for t in c])
            scores = []
            for r in rubric['tokenized']:
                r = ' '.join([t.lemma_.lower() for t in r])
                try:
                    score = self.rouge.get_scores(c, r)
                    score = score[0]['rouge-' + str(n_gram)]['r']
                except ValueError:
                    score = 0
                    print('candidate:', c, 'rubric:', r)
                    # precision = word matches / words candidate
                    # recall = word matches / words rubric item
                scores.append(score)
            labels.append(scores)
        return np.array(labels)

    def LF_rouge_L_candidate(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        labels = self._rouge(candidates, rubric, rouge_val='rouge-l')
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def LF_rouge_1_candidate(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        labels = self._rouge(candidates, rubric, rouge_val='rouge-1')
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def LF_rouge_2_candidate(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        labels = self._rouge(candidates, rubric, rouge_val='rouge-2')
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def LF_rouge_L_sentences(self, doc, rubric, lang):
        sentences, indicies = self._generate_sentences(doc)
        labels = self._rouge(sentences, rubric, rouge_val='rouge-l')
        for row, column in enumerate(np.argmax(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]

    def _rouge(self, candidates, rubric, rouge_val='rouge-l'):
        labels = []
        for c in candidates:
            c = ' '.join([t.text for t in c])
            scores = []
            for r in rubric['key_element']:
                try:
                    score = self.rouge.get_scores(c, r)
                    score = score[0][rouge_val]['f']
                except ValueError:
                    score = 0
                    print('candidate:', c, 'rubric:', r)
                    # precision = word matches / words candidate
                    # recall = word matches / words rubric item
                scores.append(score)
            labels.append(scores)
        return np.array(labels)

    def LF_edit_distance(self, doc, rubric, lang):
        labels = []
        candidates, indicies = self._generate_candidates(doc)
        for c in candidates:
            c = ' '.join([t.text for t in c])
            scores = []
            for r in rubric['tokenized']:
                r = ' '.join([t.text for t in r])
                score = edit_distance(c, r)
                scores.append(score)
            labels.append(scores)
        labels = np.array(labels)
        for row, column in enumerate(np.argmin(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] == 0:
                val = 0
            else:
                val = 1 / labels[row, column]
            yield span[0], span[1], "CUE", val

    def LF_jaccard_similarity(self, doc, rubric, lang):
        candidates, indicies = self._generate_candidates(doc)
        labels = []
        for c in candidates:
            c = [t.text.lower() for t in c]
            scores = []
            for r in rubric['tokenized']:
                r = [t.text.lower() for t in r]
                tokens1 = set(c)
                tokens2 = set(r)
                intersection = tokens1.intersection(tokens2)
                union = tokens1.union(tokens2)
                similarity = len(intersection) / len(union)
                scores.append(similarity)
            labels.append(scores)
        labels = np.array(labels)
        for row, column in enumerate(np.argmin(labels, axis=1)):
            span = indicies[row]
            if labels[row, column] > 0.0:
                yield span[0], span[1], "CUE", labels[row, column]