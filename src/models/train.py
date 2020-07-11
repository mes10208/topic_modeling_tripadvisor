import logging
import gensim
import pickle

from src.data.prepare_data import read_sample, df_to_list
from src.features.tokenize import tokenize
from src.features.dictionary import create_dictionary, term_document_matrix

def train():
	df = read_sample()
	data = df_to_list(df)
	data_lemmatized = tokenize(data)

	dictionary = create_dictionary(data_lemmatized)
	with open('data/models/dictionary.pkl', 'wb') as output_file:
		pickle.dump(dictionary, output_file)

	corpus = term_document_matrix(data_lemmatized, dictionary)

	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
		id2word=dictionary,
		num_topics=20, 
		random_state=100,
		update_every=1,
		chunksize=100,
		passes=10,
		alpha='auto',
		per_word_topics=True)

	lda_model.save("data/models/lda_model.pkl")