from typing import Generator, List

import re
import gensim

from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from spacy.lang.en import English


def clean_doc(data: List[str]) -> List[str]:
	# Remove emails
	data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

	# Remove newlines
	data = [re.sub(r'\s+', ' ', sent) for sent in data]

	# Remove quotation marks
	data = [re.sub(r"\'", "", sent) for sent in data]

	return data

def sent_to_words(sentences: List[str]) -> Generator:
	for sentence in sentences:
		# https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True elimina la puntuación

# Eliminar stopwords
def remove_stopwords(texts: List[List[str]]) -> List[List[str]]:
	stop_words = stopwords.words('english')
	stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Hacer bigrams
def make_bigrams(texts: List[List[str]]) -> List[List[str]]:
	bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	return [bigram_mod[doc] for doc in texts]

# Hacer trigrams
def make_trigrams(texts: List[List[str]]) -> List[List[str]]:
	bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
	trigram = gensim.models.Phrases(bigram[texts], threshold=100)
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	trigram_mod = gensim.models.phrases.Phraser(trigram)
	return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Lematización basada en el modelo de POS de Spacy
def lemmatization(nlp: English, texts: List[List[str]], allowed_postags: List = None) -> List[List[str]]:
	if allowed_postags is None:
		allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
	"""https://spacy.io/api/annotation"""
	texts_out = []
	for sent in texts:
		doc = nlp(" ".join(sent)) 
		texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
	return texts_out