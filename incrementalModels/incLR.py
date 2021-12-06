import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, size
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pickle

import numpy as np


def logRegression(df):

	accuracy = -1
	
	dfs = np.array(df.select('Sentiment').collect())
	dft = np.array(df.select('Tweet').collect())
	
	classifier = SGDClassifier(loss = 'log', max_iter = 1000, tol = 1e-3)
	
	clf = make_pipeline(StandardScaler(), classifier)
	clf.fit(dfs, np.ravel(dft))
	
	return accuracy
