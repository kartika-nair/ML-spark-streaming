from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession


# STREAMING

sc= SparkContext("local[2]","sent")
spark = SparkSession.builder.appName("Sentiment").getOrCreate()
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream('localhost', 6100)

def streamer(rdd):
	df=spark.read.json(rdd)
	for row in df.rdd.toLocalIterator():
		for i in range(1000):
			print(row[str(i)]['feature0'],row[str(i)]['feature1'])


# HASHING + IDF + TOKENISER -> LOGISTIC REGRESSION

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def logRegression(rdd):
	# df = spark.read.json(rdd)
	training = spark.read.format("json").load(rdd)
	lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
	lrModel = lr.fit(training)
	trainingSummary = lrModel.summary
	accuracy = trainingSummary.accuracy
	print(accuracy)

lines.foreachRDD(lambda rdd : logRegression(rdd))


# CONT STREAM
ssc.start()
ssc.awaitTermination()
