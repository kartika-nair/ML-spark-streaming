from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.clustering import GaussianMixture, GaussianMixtureModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def gmm(df):
	accuracy = -1
	
	tokenizer = Tokenizer(inputCol = 'Tweet', outputCol = 'Words')
	hashtf = HashingTF(numFeatures = 2**16, inputCol = 'Words', outputCol = 'FeatureVectors')
	idf = IDF(inputCol = 'FeatureVectors', outputCol = 'features', minDocFreq=5)
	label_stringIdx = StringIndexer(inputCol = 'Sentiment', outputCol = 'label')
	
	pipeline = Pipeline(stages = [tokenizer, hashtf, idf, label_stringIdx])
	
	subsetData = df.randomSplit([0.9, 0.1])

	pipelineFit = pipeline.fit(subsetData[0])
	train_df = pipelineFit.transform(subsetData[0])
	val_df = pipelineFit.transform(subsetData[1])
	
	gm = GaussianMixture(k = 2, maxIter = 100)
	gmModel = gm.fit(train_df)
	predictions = gmModel.transform(val_df)
	
	evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'rawPrediction')
	accuracy = evaluator.evaluate(predictions)

	return accuracy
