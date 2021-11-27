from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import StreamingLinearRegressionWithSGD
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# from sklearn.linear_model import SGDRegressor

def SGDreg(df):
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
	
	sg = StreamingLinearRegressionWithSGD(maxIter = 1000, tol = 1e-3)
	sgModel = sg.fit(train_df)
	predictions = sgModel.transform(val_df)
	
	evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'rawPrediction')
	accuracy = evaluator.evaluate(sgModel)
	
	return accuracy
