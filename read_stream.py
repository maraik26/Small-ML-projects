"""
Sample Kafka consumer, to verify that messages are coming in on the topic we expect.
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import array
from pyspark.sql.functions import udf

from pyspark.sql.types import DoubleType
# from kafka import KafkaConsumer

def calculate_regression(df):
    return(1)

topic = sys.argv[1]

spark = SparkSession \
    .builder \
    .appName("SparkStreaming") \
    .getOrCreate()

messages = spark.readStream.format('kafka') \
    .option('kafka.bootstrap.servers', '199.60.17.210:9092,199.60.17.193:9092') \
    .option('subscribe', topic).load()
lines  = messages.select(messages['value'].cast('string'))

# words = lines.select(
#    explode(
#        split(lines.value, " ")
#    ).alias("word")
# )

numbers = lines.select(
    split(lines.value, " ").alias('numbers')
)

x = udf(lambda r: float(r[0]), DoubleType())
y = udf(lambda r: float(r[1]), DoubleType())
numbers = numbers.select(x('numbers').alias('x'), y('numbers').alias('y'))

numbers = numbers \
    .withColumn('xy', numbers.x * numbers.y) \
    .withColumn('x2', numbers.x * numbers.x) \
    .withColumn('n', 0 * numbers.x + 1)

agg_data = numbers.groupBy().sum()

# print(n)
# answer = agg_data.select(
#     (agg_data['sum(xy)'] - agg_data['sum(x)'] * agg_data['sum(y)'])
# )

agg_data = agg_data \
    .withColumn('b',
        (agg_data['sum(xy)'] - agg_data['sum(x)'] * agg_data['sum(y)'] / agg_data['sum(n)']) /
        (agg_data['sum(x2)'] - (agg_data['sum(x)'] ** 2) / agg_data['sum(n)'] ))

agg_data = agg_data \
    .withColumn('a', 
        (agg_data['sum(y)'] / agg_data['sum(n)'] - agg_data['b'] * agg_data['sum(x)'] / agg_data['sum(n)']))

agg_data = agg_data.select('b', 'a', 'sum(n)')

query = agg_data \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination(600)


# print(lines.first())
# query1 = lines.first().writeStream.outputMode('append').format('console').start()


# for msg in consumer:
#     print(msg.value.decode('utf-8'))