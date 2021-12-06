import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer,VectorAssembler 
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
from sklearn.metrics import accuracy_score
import pickle
import joblib
import numpy as np

def kmm(df):
	indexer = StringIndexer(inputCol="Tweet", outputCol="Tweets_Indexed", stringOrderType='alphabetAsc')
	pipeline = Pipeline(stages=[indexer])
	pipelineFit = pipeline.fit(df)
	dataset = pipelineFit.transform(df)

	new_df=dataset.select(['Tweets_Indexed'])
	new_df_target=dataset.select(['Sentiment'])
	
	x=np.array(new_df.select('Tweets_Indexed').collect())
	y=np.array(new_df_target.select('Sentiment').collect())
	dfs_train, dfs_val, dft_train, dft_val = train_test_split(x, y, test_size = 0.1)
	model_kmm=MiniBatchKMeans(n_clusters=2, random_state=0).partial_fit(dft_train,dfs_train)
	pred = model_kmm.predict(dft_val)
	joblib.dump(model_kmm,'KMM.pkl')
	result1 = accuracy_score(dft_val,pred)
	return abs(result1)
