import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
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
import sklearn.linear_model as lm 
import pickle

import numpy as np


def logRegression(df):
	dfs = np.array(df.select('Sentiment').collect())
	dft = np.array(df.select('Tweet').collect())
	
	dfs_train, dfs_val, dft_train, dft_val = train_test_split(dfs, dft, test_size = 0.1)
	
	classifier = SGDClassifier(loss = 'log', max_iter = 1000, tol = 1e-3)
	
	clf = make_pipeline(StandardScaler(), classifier)
	clf.fit(dfs_train, np.ravel(dft_train))
	
	accuracy = clf.score(dfs_val, dft_val)
	
	return accuracy

def model(df):
	
	indexer = StringIndexer(inputCol="Tweet", outputCol="Tweets_Indexed", stringOrderType='alphabetAsc')
	pipeline = Pipeline(stages=[indexer])
	pipelineFit = pipeline.fit(df)
	dataset = pipelineFit.transform(df)

	new_df=dataset.select(['Tweets_Indexed'])
	new_df_target=dataset.select(['Sentiment'])
	
	x=np.array(new_df.select('Tweets_Indexed').collect())
	y=np.array(new_df_target.select('Sentiment').collect())
	
	model_lm=lm.LogisticRegression(warm_start=True)
	model_lm=model_lm.fit(x,y.ravel())
	
	joblib.dump(model_lm,'Logistic_Regression.pkl')
	result1=model_lm.score(x, y)
	return result1
