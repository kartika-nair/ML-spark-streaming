# MODULE IMPORTS
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score
import sys
import json
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

		# ACCURACIES
		result_lm = result_lm + model_lm.score(x,y)
		print("Logistic Regression Accuracy for batch", count, "is", model_lm.score(x,y), sep=' ')
		result_svm = result_svm + model_svm.score(x,y)
		print("SVM Accuracy for batch", count, "is", model_svm.score(x,y), sep=' ')
		result_nb = result_nb + model_nb.score(x,y)
		print("Naive Bayes Accuracy for batch", count, "is", model_nb.score(x,y), sep=' ')
		result_kmm = result_kmm + model_kmm.score(x,y)
		print("K-Means Clustering for batch", count, "is", model_kmm.score(x,y), sep=' ')
		
		# CONFUSION MATRICES
		lm_pred = model_lm.predict(x)
		print("Logistic Regression Confusion Matrix for batch", count, "is", confusion_matrix(y, lm_pred), sep=' ')
		svm_pred = model_svm.predict(x)
		print("SVM Confusion Matrix for batch", count, "is", confusion_matrix(y, svm_pred), sep=' ')
		nb_pred = model_nb.predict(x)
		print("Naive Bayes Confusion Matrix for batch", count, "is", confusion_matrix(y, nb_pred), sep=' ')
		kmm_pred = model_kmm.predict(x)
		print("K-Means Clustering Confusion Matrix for batch", count, "is", confusion_matrix(y, kmm_pred), sep=' ')
		
		# F1 SCORES
		print("Logistic Regression F1 Score for batch", count, "is", f1_score(y, lm_pred, pos_label = 4,average='micro'), sep=' ')
		print("SVM F1 Score for batch", count, "is", f1_score(y, svm_pred, pos_label = 4,average='micro'), sep=' ')
		print("Naive Bayes F1 Score for batch", count, "is", f1_score(y, nb_pred, pos_label = 4,average='micro'), sep=' ')
		print("K-Means Clustering F1 Score for batch", count, "is", f1_score(y, kmm_pred, pos_label = 4,average='micro'), sep=' ')
		
		# RECALL
		print("Logistic Regression Recall for batch", count, "is", recall_score(y, lm_pred, pos_label = 4), sep=' ')
		print("SVM Recall for batch", count, "is", recall_score(y, svm_pred, pos_label = 4), sep=' ')
		print("Naive Bayes Recall for batch", count, "is", recall_score(y, nb_pred, pos_label = 4), sep=' ')
		print("K-Means Clustering Recall for batch", count, "is", recall_score(y, kmm_pred, pos_label = 4,average='micro'), sep=' ')
		
		# PRECISION
		print("Logistic Regression Precision for batch", count, "is", precision_score(y, lm_pred, pos_label = 4), sep=' ')
		print("SVM Precision for batch", count, "is", precision_score(y, svm_pred, pos_label = 4), sep=' ')
		print("Naive Bayes Precision for batch", count, "is", precision_score(y, nb_pred, pos_label = 4), sep=' ')
		print("K-Means Clustering Precision for batch", count, "is", precision_score(y, kmm_pred, pos_label = 4,average='micro'), sep=' ')
		
		count = count + 1 

#STREAMING		
dstream = ssc.socketTextStream("localhost", 6100)

dstream1 = dstream.flatMap(lambda line: line.split("\n"))
dstream1.foreachRDD(lambda x : streamer(x))

ssc.start()
ssc.awaitTermination()


# GET AVG ACCURACIES  
acc_lm = result_lm / count
acc_svm = result_svm / count
acc_nb = result_nb / count
acc_kmm = result_kmm / count

print("Test Accuracy Logistic Regression Average: ", acc_lm * 100, sep = " ")
print("Test Accuracy SGD Average : ", acc_svm * 100, sep = " ")
print("Test Accuracy Naive Bayes Average: ", acc_nb * 100, sep = " ")
print("Test Accuracy K-Means Average: ", acc_kmm * 100, sep = " ")
