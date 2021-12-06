# MODULE IMPORTS
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score
import sys
import json
import csv
from PreProcessing import preproc
from pyspark.ml import Pipeline
import numpy as np
import pickle 
import joblib

# FILE IMPORTS
from incrementalModels import incLR, incNB, incSVM, incKM

# SPARK CONTEXT
spark = SparkSession.builder.master('local[2]').appName('Sentiment').getOrCreate()
ssc = StreamingContext(spark.sparkContext, 1)
sqlContext = SQLContext(spark)
spark.sparkContext.setLogLevel('ERROR')

# LOAD THE MODELS 
model_lm = joblib.load('Logistic_Regression.pkl')
model_svm = joblib.load('SGD.pkl')
model_nb = joblib.load('naiveBayes.pkl')
model_kmm = joblib.load('KMM.pkl')

# NUMBER OF BATCHES
count = 0 

# INITIALISE METRICS 
result_lm = 0
result_svm = 0
result_nb = 0
result_kmm = 0

# INITIALISE CSV FILES
def init_csv():
	values = ['Batch', 'Accuracy', 'F1', 'Recall', 'Precision']

	with open('metrics/logRegression.csv', 'a') as dataFile:
		writeFile = csv.writer(dataFile, dialect = 'excel')
		#for i in values:
		writeFile.writerow(values)
			
	with open('metrics/svm.csv', 'w') as dataFile:
		writeFile = csv.writer(dataFile, dialect = 'excel')
		# for i in values:
		writeFile.writerow(values)
			
	with open('metrics/nb.csv', 'w') as dataFile:
		writeFile = csv.writer(dataFile, dialect = 'excel')
		#for i in values:
		writeFile.writerow(values)

	with open('metrics/kmm.csv', 'w') as dataFile:
		writeFile = csv.writer(dataFile, dialect = 'excel')
		#for i in values:
		writeFile.writerow(values)

# METRICS FUNCTION

def streamer(rdd):

	rddValues = rdd.collect()
	
	if(len(rddValues) > 0):
		SCHEMA = ['Sentiment', 'Tweet']
		df = spark.createDataFrame(json.loads(rddValues[0]).values(), SCHEMA)
		
		df = df.na.drop()
		df = preproc(df)
		
		global result_lm
		global result_svm
		global result_nb
		global result_kmm
		global count 

		indexer = StringIndexer(inputCol = "Tweet", outputCol = "Tweets_Indexed", stringOrderType = 'alphabetAsc')
		pipeline = Pipeline(stages = [indexer])
		pipelineFit = pipeline.fit(df)
		fitted = pipelineFit.transform(df)

		df_new = fitted.select(['Tweets_Indexed'])
		df_new_target = fitted.select(['Sentiment'])

		x = np.array(df_new.select('Tweets_Indexed').collect())
		y = np.array(df_new_target.select('Sentiment').collect())
			
		# LOGISTIC REGRESSION METRICS
		values_lr = []
		
		result_lm = result_lm + model_lm.score(x,y)
		lm_pred = model_lm.predict(x)
		
		values_lr.append(count)
		values_lr.append(model_lm.score(x,y))
		values_lr.append(f1_score(y, lm_pred, pos_label = 4,average='micro'))
		values_lr.append(recall_score(y, lm_pred, pos_label = 4))
		values_lr.append(precision_score(y, lm_pred, pos_label = 4))
		
		with open('metrics/logRegression.csv', 'a') as dataFile:
			writeFile = csv.writer(dataFile, dialect = 'excel')
			writeFile.writerow(values_lr)

		# SVM METRICS
		values_svm = []
		
		result_svm = result_svm + model_svm.score(x,y)
		svm_pred = model_svm.predict(x)
		
		values_svm.append(count)
		values_svm.append(model_svm.score(x,y))
		values_svm.append(f1_score(y, svm_pred, pos_label = 4,average='micro'))
		values_svm.append(recall_score(y, svm_pred, pos_label = 4))
		values_svm.append(precision_score(y, svm_pred, pos_label = 4))
		
		with open('metrics/svm.csv', 'a') as dataFile:
			writeFile = csv.writer(dataFile, dialect = 'excel')
			writeFile.writerow(values_svm)
		
		# NAIVE BAYES METRICS
		values_nb = []
		
		result_nb = result_nb + model_nb.score(x,y)
		nb_pred = model_nb.predict(x)
		
		values_nb.append(count)
		values_nb.append(model_nb.score(x,y))
		values_nb.append(f1_score(y, nb_pred, pos_label = 4,average='micro'))
		values_nb.append(recall_score(y, nb_pred, pos_label = 4)) 
		values_nb.append(precision_score(y, nb_pred, pos_label = 4))
		
		with open('metrics/nb.csv', 'a') as dataFile:
			writeFile = csv.writer(dataFile, dialect = 'excel')
			writeFile.writerow(values_nb)
		
		# K-MEANS METRICS
		values_kmm = []
		
		result_kmm = result_kmm + model_kmm.score(x,y)
		kmm_pred = model_kmm.predict(x)
		
		values_kmm.append(count)
		values_kmm.append(f1_score(y, kmm_pred, pos_label = 4,average='micro'))
		values_kmm.append(recall_score(y, kmm_pred, pos_label = 4,average='micro'))
		values_kmm.append(precision_score(y, kmm_pred, pos_label = 4,average='micro'))
		
		with open('metrics/kmm.csv', 'a') as dataFile:
			writeFile = csv.writer(dataFile, dialect = 'excel')
			writeFile.writerow(values_kmm)
		
		
		count = count + 1 

#STREAMING		
dstream = ssc.socketTextStream("localhost", 6100)
init_csv()
dstream1 = dstream.flatMap(lambda line: line.split("\n"))
dstream1.foreachRDD(lambda x : streamer(x))

ssc.start()
ssc.awaitTermination()
