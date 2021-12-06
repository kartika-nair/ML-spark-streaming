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

import pickle
import joblib
import numpy as np
	
def model_lin_svc(df):

	indexer = StringIndexer(inputCol="Tweet", outputCol="Tweets_Indexed", stringOrderType='alphabetAsc')
	pipeline = Pipeline(stages=[indexer])
	pipelineFit = pipeline.fit(df)
	fitted = pipelineFit.transform(df)

	df_new = fitted.select(['Tweets_Indexed'])
	df_new_target = fitted.select(['Sentiment'])
	
	x=np.array(df_new.select('Tweets_Indexed').collect())
	y=np.array(df_new_target.select('Sentiment').collect())
	
	model_sgd = SGDClassifier(alpha=0.0005, loss='log', shuffle=True , max_iter=1000 , tol = 1e-3)
	
	model_sgd.partial_fit(x,y.ravel(), classes=[0,4])
	joblib.dump(model_sgd, 'SGD.pkl')
	
	result = model_sgd.score(x, y)
	return result 
