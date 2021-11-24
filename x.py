from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import StructType, StructField, StringType

# socketDF = spark.readStream.format("socket").option("host", "localhost").option("port", 6100).load()


# STREAMING

sc = SparkContext("local[2]","sent")
spark = SparkSession.builder.appName("Sentiment").getOrCreate()
ssc = StreamingContext(sc, 1)

dstream = ssc.socketTextStream('localhost', 6100)

def streamer(df, rdd):
	jsonDF = spark.read.json(rdd)	
	for row in jsonDF.rdd.toLocalIterator():
		newRow = spark.createDataFrame([(row[str(i)]['feature0'], row[str(i)]['feature1'])], columns)
		df = df.union(newRow)
	return df

columns = StructType([StructField('Sentiment', StringType(), True), StructField('Tweet', StringType(), True)])
df = spark.createDataFrame(data = [], schema = columns)

dstream.foreachRDD(lambda rdd : streamer(df, rdd))
df.show()

# CONT STREAM
ssc.start()
ssc.awaitTermination()



'''
def dataframer(rdd):
	df = spark.read.json(rdd)
	for row in df.rdd.toLocalIterator():
		if not rdd.isEmpty():
        		rdd.toDF(['Sentiment', 'Tweet']).write.save("points_json", format="json", mode="append")     

dstream.foreachRDD(lambda x : dataframer(x))

spark.sql( "select sentiment, tweet from json.`points_json`").show()



df = SQLContext.jsonRDD(RDD[dict].map(lambda rdd: json.dumps(rdd)))



schema = StructType([StructField('Sentiment', StringType(), True), StructField('Tweet', StringType(), True)])
df = spark.createDataFrame([], schema)


def dataframer():
	if count == 0:
		columns = ['Sentiment', 'Tweet']
		vals = [(row[str(i)]['feature0'], row[str(i)]['feature1'])]
		df = spark.createDataFrame(vals, columns)


def streamer(rdd):
	jsonRDD = spark.read.json(rdd)
	schema = StructType([StructField('Sentiment', StringType(), True), StructField('Tweet', StringType(), True)])
	df = spark.createDataFrame([], schema)
	i = 0
	
	for row in jsonRDD.rdd.toLocalIterator():
		newRow = spark.createDataFrame([(row[str(i)]['feature0'], row[str(i)]['feature1'])], schema)
		appended = df.union(newRow)
		i += 1


df2 = dstream.foreachRDD(lambda x : streamer(x))
df2.show()



def streamer(rdd):
	df = spark.read.json(rdd)
	for row in df.rdd.toLocalIterator():
		for i in range(1000):
			print(row[str(i)]['feature0'],row[str(i)]['feature1'])

dstream.foreachRDD(lambda x : streamer(x))


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
'''
