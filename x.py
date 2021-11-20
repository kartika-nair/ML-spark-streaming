import sys
import pyspark
from operator import add

from pyspark.sql import SparkSession, functions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: SentimentAnalysis <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

    lines = spark.read.text(sys.argv[1])
    
    # counts = lines.flatMap(lambda x: x.split(lines[1])).map(lambda x: (x, 1)).reduceByKey(add)
    
    split_col = functions.split(lines['Sentiment,Tweet'], ',')
    df = lines.withColumn('Sentiment', split_col.getItem(0)).show()
    df = lines.withColumn('Tweet', split_col.getItem(1)).show()
    
    '''
    output = lines.collect()
    for i in output:
    	print(i, sep = '\n')
    '''

    spark.stop()
