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
import joblib

import numpy as np

def logRegression(df):
	
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

	joblib.dump(model_lm, 'Logistic_Regression.pkl')
	
	result = model_lm.score(x, y)
	return result
