from typing import List

import gensim
from src.features.tokenize import tokenize
from src.features.dictionary import create_dictionary, term_document_matrix

def predict(sentences: List[str]):
	lda_model = gensim.models.ldamodel.LdaModel.load("data/models/lda_model.pkl")

	data_lemmatized = tokenize(sentences)
	dictionary = create_dictionary(data_lemmatized)
	corpus = term_document_matrix(data_lemmatized, dictionary)

	corpus_topics = []
	for bow in corpus:
		bow_topics = lda_model.get_document_topics(bow)
		topics = []
		for topic in bow_topics:
			topic_words = []
			for word in lda_model.show_topic(topic[0]):
				topic_words.append(word[0])
			topics.append({'id':str(topic[0]), 'prob':str(topic[1]), 'words':topic_words})
		corpus_topics.append({'bow':bow, 'topics':topics})

	return corpus_topics