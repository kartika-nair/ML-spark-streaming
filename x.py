from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import sys
import json

spark = SparkSession.builder.master('local[2]').appName('Sentiment').getOrCreate()
ssc = StreamingContext(spark.sparkContext, 1)
sqlContext = SQLContext(spark)
spark.sparkContext.setLogLevel('ERROR')

def logRegression(df):
	accuracy = -1
	
	df = df.na.drop()
	
	tokenizer = Tokenizer(inputCol = 'Tweet', outputCol = 'Words')
	hashtf = HashingTF(numFeatures = 2**16, inputCol = 'Words', outputCol = 'FeatureVectors')
	idf = IDF(inputCol = 'FeatureVectors', outputCol = 'features', minDocFreq=5)
	label_stringIdx = StringIndexer(inputCol = 'Sentiment', outputCol = 'label')
	
	pipeline = Pipeline(stages = [tokenizer, hashtf, idf, label_stringIdx])
	
	subsetData = df.randomSplit([0.9, 0.1])

	pipelineFit = pipeline.fit(subsetData[0])
	train_df = pipelineFit.transform(subsetData[0])
	val_df = pipelineFit.transform(subsetData[1])
	
	lr = LogisticRegression(maxIter = 100)
	lrModel = lr.fit(train_df)
	predictions = lrModel.transform(val_df)
	
	evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'rawPrediction')
	accuracy = evaluator.evaluate(predictions)

	return accuracy


def streamer(rdd):
	rddValues = rdd.collect()
	if(len(rddValues) > 0):
		SCHEMA = ['Sentiment', 'Tweet']
		df = spark.createDataFrame(json.loads(rddValues[0]).values(), SCHEMA)
		# df.show(truncate = False)
		accuracy_logRegression = logRegression(df)
		print('Logistic Regression Accuracy =', accuracy_logRegression)

dstream = ssc.socketTextStream("localhost", 6100)

dstream1 = dstream.flatMap(lambda line: line.split("\n"))
dstream1.foreachRDD(lambda x : streamer(x))

ssc.start()
ssc.awaitTermination()
