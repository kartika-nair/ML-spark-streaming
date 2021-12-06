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
import pickle

import numpy as np


def logRegression(df):
	
	dfs = np.array(df.select('Sentiment').collect())
	dft = np.array(df.select('Tweet').collect())
	
	le = preprocessing.LabelEncoder()
	dft = le.fit_transform(np.array(df.select('Tweet').collect()))
	
	dfs_train, dfs_val, dft_train, dft_val = train_test_split(dfs, dft, test_size = 0.1)
	sgd = SGDClassifier()
	dfs_train.reshape(-1,1)
	dft_train.reshape(-1,1)
	sgd.partial_fit(dft_train, dfs_train, classes = [0,4])
	pred_s = sgd.predict(dft_val)
	
	accuracy = accuracy_score( pred_s,dfs_val)
	
	return accuracy
