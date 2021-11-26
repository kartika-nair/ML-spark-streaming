from pyspark import rdd
from pyspark.sql.types import StructType, StructField, StringType

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# STREAMING

sc = SparkContext("local[2]", "sent")
spark = SparkSession.builder.appName("Sentiment").getOrCreate()
spark.sparkContext.setLogLevel('WARN')
ssc = StreamingContext(sc, 1)

dstream = ssc.socketTextStream('localhost', 6100)

def logRegression(df):
	accuracy = -1
	
	df = df.na.drop()
	
	tokenizer = Tokenizer(inputCol = 'Tweet', outputCol = 'Words')
	hashtf = HashingTF(numFeatures = 2**16, inputCol = 'Words', outputCol = 'FeatureVectors')
	idf = IDF(inputCol = 'FeatureVectors', outputCol = 'Features', minDocFreq=5)
	label_stringIdx = StringIndexer(inputCol = 'Sentiment', outputCol = 'Label')
	
	pipeline = Pipeline(stages = [tokenizer, hashtf, idf, label_stringIdx])
	
	subsetData = df.randomSplit([0.9, 0.1])

	pipelineFit = pipeline.fit(subsetData[0])
	train_df = pipelineFit.transform(subsetData[0])
	val_df = pipelineFit.transform(subsetData[1])
	# train_df.show(5)
	
	lr = LogisticRegression(maxIter = 100)
	lrModel = lr.fit(train_df)
	predictions = lrModel.transform(val_df)
	
	evaluator = BinaryClassificationEvaluator(rawPredictionCol = "RawPrediction")
	accuracy = evaluator.evaluate(predictions)

	return accuracy


def streamer(rdd):
	jsonDF = spark.read.json(rdd)
	schema = StructType([StructField('Sentiment', StringType(), True), StructField('Tweet', StringType(), True)])
	df = spark.createDataFrame([], schema)
	
	for row in jsonDF.rdd.toLocalIterator():
		for i in range(10000):
			# print(row[str(i)]['feature0'],row[str(i)]['feature1'])
			newRow = spark.createDataFrame([(row[str(i)]['feature0'], row[str(i)]['feature1'])], schema)
			df = df.unionByName(newRow)
			
	accuracy_logRegression = logRegression(df)
	print('Logistic Regression Accuracy =', accuracy_logRegression)

dstream.foreachRDD(lambda rdd : streamer(rdd))

# CONT STREAM
ssc.start()
ssc.awaitTermination()
