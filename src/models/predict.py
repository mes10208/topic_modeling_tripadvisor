from typing import List
import pickle

import gensim
from src.features.tokenize import tokenize
from src.features.dictionary import create_dictionary, term_document_matrix

def predict(sentences: List[str]):
	with open('data/models/dictionary.pkl', 'rb') as input_file:
		dictionary = pickle.load(input_file)
	
	lda_model = gensim.models.ldamodel.LdaModel.load("data/models/lda_model.pkl")

	data_lemmatized = tokenize(sentences)
	corpus = term_document_matrix(data_lemmatized, dictionary)

	corpus_topics = []
	for bow in corpus:
		bow_topics = lda_model[bow][0]
		topics = []
		for topic in bow_topics:
			topic_words = []
			for word in lda_model.show_topic(topic[0]):
				topic_words.append(word[0])
			topics.append({'id':str(topic[0]), 'prob':str(topic[1]), 'words':topic_words})
		corpus_topics.append({'bow':bow, 'topics':topics})

	return corpus_topics