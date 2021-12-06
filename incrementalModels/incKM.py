import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from pyspark.ml.evaluation import ClusteringEvaluator

import numpy as np


def kmm(df):
	
	accuracy = -1
	
	df1 = np.array(df.collect())
	df_sent = np.array(df.select('Sentiment').collect())
	
	le = LabelEncoder()
	df_twt = le.fit_transform(np.array(df.select('Tweet').collect()))
	
	sent_train, sent_test, twt_train, twt_test = train_test_split(df_sent, df_twt, test_size = 0.1, random_state = 0)
	
	kmeans = MiniBatchKMeans(n_clusters = 10)
	model = kmeans.partial_fit(sent_train, np.ravel(twt_train))
	
	predictions = kmeans.predict(sent_test)
	
	accuracy = accuracy_score(np.ravel(twt_test), predictions)
	
	return accuracy
'''import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from pyspark.ml.evaluation import ClusteringEvaluator

import numpy as np


def kmm(df):
	
	accuracy = -1
	
	df1 = np.array(df.collect())
	df_sent = np.array(df.select('Sentiment').collect())
	
	le = LabelEncoder()
	df_twt = le.fit_transform(np.array(df.select('Tweet').collect()))
	
	sent_train, sent_test, twt_train, twt_test = train_test_split(df_sent, df_twt, test_size = 0.1, random_state = 0)
	
	kmeans = MiniBatchKMeans(n_clusters = 10)
	y = twt_train.ravel()
	train_y = np.array(y).astype(int)
	model = kmeans.partial_fit(sent_train, train_y)
	
	predictions = kmeans.predict(sent_test)
	p = predictions.ravel()
	pr = np.array(p).astype(int)
	accuracy = accuracy_score(twt_test, p)
	
	return accuracy'''
