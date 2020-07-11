from typing import List

from fastapi import FastAPI, Query

from src.models.train import train
from src.models.predict import predict

app = FastAPI()


@app.get('/')
async def test():
	return {'return':'hello wolrd'}

@app.post('/train')
async def train_model():
	train()

	return {'Result': 'model.pkl produced'}

@app.get('/predict')
async def predict_sentences_topic(sentences: List[str] = Query(..., description='Sentences to process')):
	topics = predict(sentences)

	return {'Result': topics}