import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.utils import column_or_1d

import numpy as np

def nBayes(df):
	
	bern = BernoulliNB()
	accuracy = -1
	
	x = np.asarray(df.select('Sentiment').collect())
	y = np.asarray(df.select('Tweet').collect())
	y = column_or_1d(y)
	
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 13)
	
	bern.partial_fit(X_train, y_train, classes = [0, 4])
	predictions = bern.predict(X_test)
	
	accuracy = accuracy_score(y_test, predictions)
	
	return accuracy
