# IMPORTS

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

import sys
import json

# FILE IMPORTS
from incrementalModels import incLR, incNB, incSVM, incKM
from PreProcessing import preproc
from pyspark.ml import Pipeline
import numpy as np
import pickle 
import joblib

spark = SparkSession.builder.master('local[2]').appName('Sentiment').getOrCreate()
ssc = StreamingContext(spark.sparkContext, 1)
sqlContext = SQLContext(spark)
spark.sparkContext.setLogLevel('ERROR')

#load the models 
model_lm = joblib.load('Logistic_Regression.pkl')
# model_svm = joblib.load('SGD.pkl')
# model_nb = joblib.load('naiveBayes.pkl')
# model_kmm = joblib.load('kmm.pkl')

# number of batches 
count = 0 
# initilaise the metrics 
result_lm = 0
result_svm = 0
result_nb = 0
result_kmm = 0

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

		
		indexer = StringIndexer(inputCol="Tweet", outputCol="Tweets_Indexed", stringOrderType='alphabetAsc')
		pipeline = Pipeline(stages=[indexer])
		pipelineFit = pipeline.fit(df)
		fitted = pipelineFit.transform(df)

		df_new = fitted.select(['Tweets_Indexed'])
		df_new_target = fitted.select(['Sentiment'])

		x=np.array(df_new.select('Tweets_Indexed').collect())
		y=np.array(df_new_target.select('Sentiment').collect())

		result_lm = result_lm + model_lm.score(x,y)
		print("Logistic Regression Accuracy for batch",count,"is",result_lm,sep=' ')
		result_svm = result_svm + model_svm.score(x,y)
		print("SVM Accuracy for batch",count,"is",result_lm,sep=' ')
		result_nb = result_nb + model_nb.score(x,y)
		print("Naive Bayes Accuracy for batch",count,"is",result_lm,sep=' ')
		
		count = count + 1 
		
dstream = ssc.socketTextStream("localhost", 6100)

dstream1 = dstream.flatMap(lambda line: line.split("\n"))
dstream1.foreachRDD(lambda x : streamer(x))

ssc.start()
ssc.awaitTermination()

# Get average accuracies  
acc_lm = result_lm / count
acc_svm = result_svm / count
acc_nb = result_nb / count

print("Test Accuracy LogRegression Average: ", acc_lm*100 , sep=" ")
print("Test Accuracy SGD Average : ", acc_svm*100 , sep=" ")
print("Test Accuracy Naive Bayes Average: ", acc_nb*100 , sep=" ")

