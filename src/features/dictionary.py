from typing import List, Tuple

from gensim.corpora import Dictionary


def create_dictionary(data_lemmatized: List[List[str]]):
	return Dictionary(data_lemmatized)


def term_document_matrix(texts, dictionary: Dictionary) -> List[List[Tuple[int, int]]]:
	return [dictionary.doc2bow(text) for text in texts]