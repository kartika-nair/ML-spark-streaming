from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from org.apache.spark.ml.evaluation import ClusteringEvaluator

def kmm(df):
	accuracy = -1
	
	tokenizer = Tokenizer(inputCol = 'Tweet', outputCol = 'Words')
	hashtf = HashingTF(numFeatures = 2**16, inputCol = 'Words', outputCol = 'FeatureVectors')
	idf = IDF(inputCol = 'FeatureVectors', outputCol = 'features', minDocFreq=5)
	label_stringIdx = StringIndexer(inputCol = 'Sentiment', outputCol = 'label')
	
	pipeline = Pipeline(stages = [tokenizer, hashtf, idf, label_stringIdx])

	pipelineFit = pipeline.fit('Sentiment', 'Tweet')
	train_df = pipelineFit.transform(df[1:])
	# val_df = pipelineFit.transform(subsetData[1])
	
	kmeans = KMeans().setK(10).setSeed(1)
	model = kmeans.fit(train_df)
	centers = model.clusterCenters()
	
	squaredError = model.computeCost(train_df)

	return squaredError
