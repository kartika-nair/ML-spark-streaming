import sys
import pyspark
from operator import add
from pyspark.sql import SparkSession, functions
import stream

import findspark
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext

sc = SparkContext("local[2]", "NetworkWordCount")
spark = SparkSession.builder.appName("python spark create rdd").config("spark.some.config.option","some-value").getOrCreate()
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)

lines = ssc.socketTextStream("localhost", 6100)

# lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
words = lines.flatMap(lambda line: line.split("\n"))

words.pprint()

pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)
# wordCounts.pprint()

ssc.start()
ssc.awaitTermination()

'''

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: SentimentAnalysis <file>", file=sys.stderr)
        sys.exit(-1)
  
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

    lines = spark.read.text(sys.argv[1])
    sent, tw = lines.flatMap(lambda x: x.split(lines[1]))
    
    for i in sent.collect():
    	print(i)
    
    output = lines.collect()
    for i in output:
    	print(i, sep = '\n')

    spark.stop()
'''
