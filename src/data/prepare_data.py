from typing import List

import pandas as pd
from pandas import DataFrame

from nltk.corpus import stopwords

def read_sample() -> DataFrame:
	df = pd.read_json('data/raw/newsgroups.json')
	##df['rating'] = df['rating'].astype(dtype='int64')

	return df

def df_to_list(df: DataFrame) -> List[str]:
	data = df.content.values.tolist()

	return data