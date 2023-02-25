from typing import (Callable, Collection, Dict, Iterable, Optional, Sequence, Set, Tuple)

import numpy as np
import pandas as pd
import skweak
from spacy.tokens import Doc, Span, Token  # type: ignore
from skweak import base, aggregation, utils
from tqdm import tqdm
import myutils
import config
from weak_supervision import WeakSupervisionSoft
from abc import abstractmethod


class SpanAnnotator(base.AbstractAnnotator):
    """Generic class for the annotation of token spans"""

    def __init__(self, name: str):
        """Initialises the annotator with a source name"""

        super(SpanAnnotator, self).__init__(name)

        # Set of other labelling sources that have priority
        self.incompatible_sources = []

    # type:ignore
    def add_incompatible_sources(self, other_sources: Sequence[str]):
        """Specifies a list of sources that are not compatible with the current
        source and should take precedence over it in case of overlap"""

        self.incompatible_sources.extend(other_sources)

    def __call__(self, doc: Doc, rubric: pd.DataFrame, lang: str) -> Doc:

        # We start by clearing all existing annotations
        doc.spans[self.name] = []

        # And we look at all suggested spans
        for start, end, label, prob in self.find_spans(doc, rubric, lang):

            # We only add the span if it is compatible with other sources
            if self._is_allowed_span(doc, start, end):
                span = Span(doc, start, end, label)
                doc.spans[self.name].append(span)

        return doc

    @abstractmethod
    def find_spans(self, doc: Doc, rubric: pd.DataFrame, lang: str) -> Iterable[Tuple[int, int, str, float]]:
        """Generates (start, end, label) triplets corresponding to token-level
        spans associated with a given label. """

        raise NotImplementedError("Must implement find_spans method")

    def _is_allowed_span(self, doc, start, end):
        """Checks whether the span is allowed (given incompatibilities with other sources)"""

        for other_source in self.incompatible_sources:

            intervals = sorted((span.start, span.end) for span in
                               doc.spans.get(other_source, []))

            # Performs a binary search to efficiently detect overlapping spans
            start_search, end_search = utils._binary_search(
                start, end, intervals)
            for interval_start, interval_end in intervals[start_search:end_search]:
                if start < interval_end and end > interval_start:
                    return False
        return True


class RubricAnnotator(SpanAnnotator):
    """Annotation based on a heuristic function that generates (start,end,label)
    given a spacy document"""

    def __init__(self, name: str,
                 function: Callable[[Doc, pd.DataFrame, str], Iterable[Tuple[int, int, str, float]]],
                 to_exclude: Sequence[str] = ()):
        """Create an annotator based on a function generating labelled spans given
        a Spacy Doc object. Spans that overlap with existing spans from sources
        listed in 'to_exclude' are ignored. """

        super(RubricAnnotator, self).__init__(name)
        self.find_spans = function
        # self.add_incompatible_sources(to_exclude)


class WeakSupervisionHMM:
    def __init__(self, rubrics=None, meteor_th=0.25, ngram_th=0.5, rouge_th=0.5, edit_dist_th=0.5, paraphrase_th=0.5,
                 bleu_th=0.5, jaccard_th=0.5, mode='hmm'):
        self.hmm = aggregation.HMM("hmm", ["CUE"], prefixes='IO')
        self.voter = skweak.voting.SequentialMajorityVoter("maj_voter", labels=["CUE"], prefixes='IO')
        self.mode = mode
        self.rubrics = rubrics
        self.annotated_data = []
        self.PARAPHRASE_THRESHOLD = paraphrase_th
        self.METEOR_THRESHOLD = meteor_th
        self.NGRAM_THRESHOLD = ngram_th
        self.ROUGE_THRESHOLD = rouge_th
        self.BLEU_THRESHOLD = bleu_th
        self.JACCARD_THRESHOLD = jaccard_th
        self.EDIT_DISTANCE_THRESHOLD = edit_dist_th
        self.ws = WeakSupervisionSoft(self.rubrics)
        self.labeling_functions = [
            RubricAnnotator('LF_meteor_sentences', self.LF_meteor_sentences),
            RubricAnnotator('LF_meteor_candidates', self.LF_meteor_candidates),
            RubricAnnotator('LF_noun_phrases', self.LF_noun_phrases),
            RubricAnnotator('LF_lemma_match', self.LF_lemma_match),
            RubricAnnotator('LF_pos_match', self.LF_pos_match),
            RubricAnnotator('LF_shape_match', self.LF_shape_match),
            RubricAnnotator('LF_stem_match', self.LF_stem_match),
            RubricAnnotator("LF_lemma_match_without_stopwords", self.LF_lemma_match_without_stopwords),
            RubricAnnotator("LF_meteor_sentences", self.LF_meteor_sentences),
            RubricAnnotator("LF_dep_match", self.LF_dep_match),
            RubricAnnotator('LF_dep_match_without_stopwords', self.LF_dep_match_without_stopwords),
            RubricAnnotator('LF_bi_gram_overlap', self.LF_bi_gram_overlap),
            RubricAnnotator('LF_tri_gram_overlap', self.LF_tri_gram_overlap),
            RubricAnnotator('LF_tetra_gram_overlap', self.LF_tetra_gram_overlap),
            RubricAnnotator('LF_penta_gram_overlap', self.LF_penta_gram_overlap),
            RubricAnnotator('LF_rouge_L_candidate', self.LF_rouge_L_candidate),
            RubricAnnotator('LF_rouge_1_candidate', self.LF_rouge_1_candidate),
            RubricAnnotator('LF_rouge_2_candidate', self.LF_rouge_2_candidate),
            RubricAnnotator('LF_rouge_L_sentences', self.LF_rouge_L_sentences),
            RubricAnnotator('LF_word_alignment', self.LF_word_alignment),
            RubricAnnotator('LF_edit_distance', self.LF_edit_distance),
            RubricAnnotator('LF_paraphrase_detection_sentences', self.LF_paraphrase_detection_sentences),
            RubricAnnotator('LF_paraphrase_detection_candidates', self.LF_paraphrase_detection_candidates),
            RubricAnnotator('LF_bleu_candidates', self.LF_bleu_candidates),
            RubricAnnotator('LF_jaccard_similarity', self.LF_jaccard_similarity),
        ]

    def _get_rubric(self, question_id: str):
        return self.rubrics[question_id]

    def _get_spans_above_threshold(self, spans, threshold):
        spans = list(spans)
        spans = [s for s in spans if s[3] >= threshold]
        yield from spans

    def LF_dep_match_without_stopwords(self, doc, rubric, lang):
        return self.ws.LF_dep_match_without_stopwords(doc, rubric, lang)

    def LF_dep_match(self, doc, rubric, lang):
        return self.ws.LF_dep_match(doc, rubric, lang)

    def LF_lemma_match_without_stopwords(self, doc, rubric, lang):
        return self.ws.LF_lemma_match_without_stopwords(doc, rubric, lang)

    def LF_pos_match_without_stopwords(self, doc, rubric, lang):
        return self.ws.LF_pos_match_without_stopwords(doc, rubric, lang)

    def LF_noun_phrases(self, doc, rubric, lang):
        return self.ws.LF_noun_phrases(doc, rubric, lang)

    def LF_meteor_candidates(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_meteor_candidates(doc, rubric, lang), self.METEOR_THRESHOLD)

    def LF_meteor_sentences(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_meteor_sentences(doc, rubric, lang), self.METEOR_THRESHOLD)

    def LF_stem_match(self, doc, rubric, lang):
        return self.ws.LF_stem_match(doc, rubric, lang)

    def LF_lemma_match_without_stopwords(self, doc, rubric, lang):
        return self.ws.LF_lemma_match_without_stopwords(doc, rubric, lang)

    def LF_lemma_match(self, doc, rubric, lang):
        return self.ws.LF_lemma_match(doc, rubric, lang)

    def LF_pos_match(self, doc, rubric, lang):
        return self.ws.LF_pos_match(doc, rubric, lang)

    def LF_shape_match(self, doc, rubric, lang):
        return self.ws.LF_shape_match(doc, rubric, lang)

    def LF_tag_match(self, doc, rubric, lang):
        return self.ws.LF_tag_match(doc, rubric, lang)

    def LF_word_alignment(self, doc, rubric, lang):
        return self.ws.LF_word_alignment(doc, rubric, lang)

    def LF_bi_gram_overlap(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_bi_gram_overlap(doc, rubric, lang), self.NGRAM_THRESHOLD)

    def LF_tri_gram_overlap(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_tri_gram_overlap(doc, rubric, lang), self.NGRAM_THRESHOLD)

    def LF_tetra_gram_overlap(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_tetra_gram_overlap(doc, rubric, lang), self.NGRAM_THRESHOLD)

    def LF_penta_gram_overlap(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_penta_gram_overlap(doc, rubric, lang), self.NGRAM_THRESHOLD)

    def LF_rouge_L_candidate(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_rouge_L_candidate(doc, rubric, lang), self.ROUGE_THRESHOLD)

    def LF_rouge_1_candidate(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_rouge_1_candidate(doc, rubric, lang), self.ROUGE_THRESHOLD)

    def LF_rouge_2_candidate(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_rouge_2_candidate(doc, rubric, lang), self.ROUGE_THRESHOLD)

    def LF_rouge_L_sentences(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_rouge_L_sentences(doc, rubric, lang), self.ROUGE_THRESHOLD)

    def LF_edit_distance(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_edit_distance(doc, rubric, lang), self.EDIT_DISTANCE_THRESHOLD)

    def LF_paraphrase_detection_sentences(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_paraphrase_detection_sentences(doc, rubric, lang), self.PARAPHRASE_THRESHOLD)

    def LF_paraphrase_detection_candidates(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_paraphrase_detection_candidates(doc, rubric, lang), self.PARAPHRASE_THRESHOLD)

    def LF_bleu_candidates(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_bleu_candidates(doc, rubric, lang), self.BLEU_THRESHOLD)

    def LF_jaccard_similarity(self, doc, rubric, lang):
        return self._get_spans_above_threshold(self.ws.LF_jaccard_similarity(doc, rubric, lang), self.JACCARD_THRESHOLD)

    def predict(self, data):
        docs = []
        for i, d in tqdm(data.iterrows()):
            question_id = d['question_id']
            rubric = self._get_rubric(question_id)
            doc = d['tokenized']
            lang = d['lang']
            for lf in self.labeling_functions:
                doc = lf(doc, rubric, lang)
            docs.append(doc)
            if self.mode == 'voter':
                prediction = self.voter.pipe(docs)
            elif self.mode == 'hmm':
                prediction = self.hmm.pipe(docs)
        return list(prediction)

    def fit(self, data):
        # use rubric as gazetteer
        docs = []
        for i, d in tqdm(data.iterrows()):
            question_id = d['question_id']
            rubric = self._get_rubric(question_id)
            doc = d['tokenized']
            lang = d['lang']
            for lf in self.labeling_functions:
                doc = lf(doc, rubric, lang)
            docs.append(doc)
        # aggregation model
        if self.mode == 'voter':
            self.annotated_data = list(self.voter.pipe(docs))
        elif self.mode == 'hmm':
            self.annotated_data = self.hmm.fit_and_aggregate(docs)
        # for more than 4 em steps it's need a custom implementation of the aggregator model. n_iter is hardcoded to 4
        #self.annotated_data = self.hmm.fit_and_aggregate(docs)

        # self.annotated_data = self.hmm.pipe(docs)
        return self.annotated_data


def main():
    # Read data
    X_train = pd.read_json(config.PATH_DATA + '/' + 'training_dataset.json')[0:10]
    X_dev = pd.read_json(config.PATH_DATA + '/' + 'dev_dataset.json')[0:10]
    X_train = myutils.tokenize_data(X_train)
    X_dev = myutils.tokenize_data(X_dev)
    rubrics = myutils.load_rubrics(config.PATH_RUBRIC)
    rubrics = myutils.prepare_rubrics(rubrics)

    th = 0.5
    ws = WeakSupervisionHMM(rubrics=rubrics, meteor_th=th, ngram_th=th, rouge_th=th, edit_dist_th=th,
                            paraphrase_th=th, bleu_th=th, jaccard_th=th, mode='voter')


    train_result = ws.fit(X_train)
    dev_result = ws.predict(X_dev)

    myutils.save_annotated_corpus(train_result, "corpora/train_labeled_data_hmm.spacy")
    myutils.save_annotated_corpus(dev_result, "corpora/dev_labeled_data_hmm.spacy")
    # create the json file from it but without the detailed LF annotations
    for j, data in enumerate([X_train, X_dev]):
        annotated_data = []
        for i, d in tqdm(data.iterrows()):
            item = {
                'lang': d['lang'],
                'question_id': d['question_id'],
                'question': d['question'],
                'reference_answer': d['reference_answer'],
                'score': d['score'],
                'label': d['label'],
                'student_answer': d['student_answer'],
                'labeling_functions': {},
            }
            annotated_data.append(item)
        if j == 0:
            file_name = 'training_ws'
        else:
            file_name = 'dev_ws'
        myutils.save_json(annotated_data, config.PATH_DATA, file_name + '_hmm' + '.json')

if __name__ == "__main__":
    main()