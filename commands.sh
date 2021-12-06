# Command to run stream.py file
python3 stream.py -f sentiment -b <batch_size>

# Command to run either main.py or incrementalMain.py
$SPARK_HOME/bin/spark-submit <file>.py

# Keep train and test CSV files in subdirectory titled 'sentiment'
