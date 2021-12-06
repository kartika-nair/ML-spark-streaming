# IMPORTS

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

import sys
import json

# FILE IMPORTS
from incrementalModels import incLR, incNB, incSVM, incKM
from PreProcessing import preproc

spark = SparkSession.builder.master('local[2]').appName('Sentiment').getOrCreate()
ssc = StreamingContext(spark.sparkContext, 1)
sqlContext = SQLContext(spark)
spark.sparkContext.setLogLevel('ERROR')

def streamer(rdd):
	rddValues = rdd.collect()
	if(len(rddValues) > 0):
		SCHEMA = ['Sentiment', 'Tweet']
		df = spark.createDataFrame(json.loads(rddValues[0]).values(), SCHEMA)
		
		df = df.na.drop()
		df = preproc(df)
		
		accuracy_logRegression = incLR.logRegression(df)
		print('Logistic Regression Accuracy =', accuracy_logRegression)
		
		'''
		accuracy_NB = incNB.nBayes(df)
		print('Naive Bayes Accuracy =', accuracy_NB)
		
		accuracy_SVM = incSVM.linSVC(df)
		print('Linear SVM Accuracy =', accuracy_SVM)
		
		error_KMM = incKM.kmm(df)
		print('K-Means Clustering Squared Error =', error_KMM)
		'''

dstream = ssc.socketTextStream("localhost", 6100)

dstream1 = dstream.flatMap(lambda line: line.split("\n"))
dstream1.foreachRDD(lambda x : streamer(x))

ssc.start()
ssc.awaitTermination()
