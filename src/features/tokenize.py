from typing import List

from src.features import nlp
from src.features.utils import clean_doc, sent_to_words, remove_stopwords, make_bigrams, lemmatization


def tokenize(documents: List[str]) -> List[List[str]]:

	documents = clean_doc(documents)
	document_words = list(sent_to_words(documents))
	document_words = remove_stopwords(document_words)
	document_words = make_bigrams(document_words)
	document_words = lemmatization(nlp, document_words)

	return document_words