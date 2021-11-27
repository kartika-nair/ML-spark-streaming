from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import *

def preproc(reg_df):
	reg_df0 = reg_df.withColumn("Tweet", regexp_replace("Tweet","@[A-Za-z0-9]+","")) 
	reg_df1 = reg_df0.withColumn("Tweet", trim(col("Tweet")))
	reg_df2 = reg_df1.withColumn("Tweet", regexp_replace("Tweet","https?://[A-Za-z0-9./]+",""))
	reg_df3 = reg_df2.withColumn("Tweet", regexp_replace("Tweet","[^a-zA-Z0-9 ]"," ")) 
	reg_df4 = reg_df3.withColumn("Tweet", regexp_replace("Tweet","[^\w\s]",""))
	
	return reg_df4
