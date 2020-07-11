from gensim.models import CoherenceModel
from src.models.train import train

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
%matplotlib inline


def visualize():
	lda_model, corpus, data_lemmatized, dictionary = train()

	#Perplejidad
	print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

	# Score de coherencia
	coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)

	# Visualizamos los temas
	pyLDAvis.enable_notebook()
	vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
	vis