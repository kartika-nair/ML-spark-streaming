import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, size
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sklearn import preprocessing 
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer,VectorAssembler #OneHotEncoderEstimator
import sklearn.naive_bayes as nb
import pickle
import joblib

import numpy as np


def nBayes(df):
	
	indexer = StringIndexer(inputCol = "Tweet", outputCol = "Tweets_Indexed", stringOrderType = 'alphabetAsc')
	pipeline = Pipeline(stages = [indexer])
	pipelineFit = pipeline.fit(df)
	dataset = pipelineFit.transform(df)

	new_df = dataset.select(['Tweets_Indexed'])
	new_df_target = dataset.select(['Sentiment'])
	
	twt = np.array(new_df.select('Tweets_Indexed').collect())
	sent = np.array(new_df_target.select('Sentiment').collect())
	
	model_nb = BernoulliNB()
	model_nb = model_nb.fit(twt, sent.ravel())
	
	joblib.dump(model_nb,'naiveBayes.pkl')
	acc = model_nb.score(twt, sent)
	return acc
