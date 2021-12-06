import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer,VectorAssembler 
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
import pickle
import joblib
import numpy as np

def models(df):
	indexer = StringIndexer(inputCol="Tweet", outputCol="Tweets_Indexed", stringOrderType='alphabetAsc')
	pipeline = Pipeline(stages=[indexer])
	pipelineFit = pipeline.fit(df)
	dataset = pipelineFit.transform(df)

	new_df=dataset.select(['Tweets_Indexed'])
	new_df_target=dataset.select(['Sentiment'])
	
	x=np.array(new_df.select('Tweets_Indexed').collect())
	y=np.array(new_df_target.select('Sentiment').collect())
	
	model_kmm=KMeans(n_clusters=1000, random_state=0).fit(x,y)

	joblib.dump(model_kmm,'KMM.pkl')
	result1 = model_kmm.score(x,y.ravel())
	return result1
