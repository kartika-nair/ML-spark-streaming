import findspark
# findspark.init()
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

sc = SparkContext(appName="SA")
spark = SparkSession.builder.appName("python spark create rdd").config("spark.some.config.option","some-value").getOrCreate()
# Create a local StreamingContext with batch interval of 1 second
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)
# Create a DStream that conencts to hostname:port
lines = ssc.socketTextStream("localhost", 6100)
ssc.start()
ssc.awaitTermination()

