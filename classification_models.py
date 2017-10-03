#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:49:33 2017

To use cluster it is necessary config hdfs with file
From: https://stackoverflow.com/questions/27299923/how-to-load-local-file-in-sc-textfile-instead-of-hdfs

Run command: pyspark --packages com.databricks:spark-csv_2.10:1.1.0

@author: diego
"""
from __future__ import print_function

from pyspark.sql import SQLContext
from pyspark.sql.types import *

import numpy as np
import unicodedata

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, StringIndexer
from pyspark.sql.functions import *    

from pyspark.sql import functions as func


if __name__ == "__main__":    
    
sqlContext = SQLContext(sc)

#news_complete_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', delimiter=';').load('file:///home/lara/diegomarques/cod_clas_con_res_tit_semIne.csv')
news_complete_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', delimiter=';').load('hdfs://cluster0.local:8020/hdfs/diegomarques/data/cod_clas_con_res_tit_semIne.csv')

news_df = news_complete_data

# Spark 2.2.0 Upoad File into data frame
#news_complete_data = spark.read.format("csv").option("header", "true").option("inferSchema",
#                           "true").option("delimiter", ";").load("cod_clas_con_res_tit_semIne.csv")
news_df = news_complete_data.sample(False, 0.50, 1234L)

# Remove extra spaces
news_df = news_df.withColumn("classificacao", trim(news_df.classificacao))
news_df = news_df.withColumn("conteudo", trim(news_df.conteudo))
news_df = news_df.withColumn("resumo", trim(news_df.resumo))
news_df = news_df.withColumn("titulo", trim(news_df.titulo))
	
# Clean everything inside <>    
news_df = news_df.withColumn('conteudo', regexp_replace('conteudo', '<.*?>', ''))
news_df = news_df.withColumn('resumo', regexp_replace('resumo', '<.*?>', ''))
news_df = news_df.withColumn('titulo', regexp_replace('titulo', '<.*?>', ''))

##############################
# Clean text removing accents
# Receives  a string as an argument
def remove_accents_(inputStr):
    # first, normalize strings:
    nfkdStr = unicodedata.normalize('NFKD', inputStr)
    # Keep chars that has no other char combined (i.e. accents chars)
    withOutAccents = u"".join([c for c in nfkdStr if not unicodedata.combining(c)])
    return withOutAccents

remove_accents = udf(lambda c: remove_accents_(c) if c != None else c, StringType())

news_df = news_df.withColumn('classificacao', remove_accents(news_df.classificacao))
news_df = news_df.withColumn('conteudo', remove_accents(news_df.conteudo))
news_df = news_df.withColumn('resumo', remove_accents(news_df.resumo))
news_df = news_df.withColumn('titulo', remove_accents(news_df.titulo))

# Null categories
# Get only conteudo not empty
news_df = news_df.where(news_df.conteudo != '')
# Total 100% = 709538
# Total 50% = 354452

# Calculate variance
news_df_group = news_df.groupBy('classificacao').count().sort(col("count").desc())
news_df_group = news_df_group.withColumn('count', news_df_group['count'].cast(FloatType()))
news_df_group = news_df_group.select(col('count'))
group_variance = news_df_group.agg(func.avg('count').alias('avg_variance')).rdd
group_variance_df = sqlContext.createDataFrame(group_variance, news_df_group.schema)

# Total
#+---------+
#|    count|
#+---------+
#|24466.828| 156,418
#+---------+

# 50%
#+---------+
#|    count|
#+---------+
#|12222.482| 110,555
#+---------+


###################################################
# Get features from dataframe and transform in a vector  

data = news_df

# StringIndex
str_idx_model = StringIndexer(inputCol="classificacao", outputCol="idx_classificacao").fit(data)
data_idx_clas = str_idx_model.transform(data)

# Tokenize
tk_model = Tokenizer(inputCol="conteudo", outputCol="tk_conteudo")
data_tk_cont = tk_model.transform(data_idx_clas)

from pyspark.ml.feature import CountVectorizer
data_vectorizer = CountVectorizer(inputCol="tk_conteudo", outputCol="vectorizer_conteudo").fit(data_tk_cont)
len(data_vectorizer.vocabulary)
#262144

# N-gram
from pyspark.ml.feature import NGram
ngram = NGram(n=3, inputCol="tk_conteudo", outputCol="ngrams")
data_ngram = ngram.transform(data_tk_cont)

# TF-IDF
hashingTF = HashingTF(inputCol="tk_conteudo", outputCol="v_conteudo", numFeatures=50000)
data_v_cont = hashingTF.transform(data_tk_cont)

idf = IDF(inputCol="v_conteudo", outputCol="features_conteudo")
idfModel = idf.fit(data_v_cont)
data_idf_cont = idfModel.transform(data_v_cont)
    
    ###################################################
    # Naive Bayes

    # Spark 2.2.0
    #from pyspark.ml.classification import NaiveBayes
    #from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    #Spark 1.6.0
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint  

from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
    
data_to_test = data_idf_cont.select(col("idx_classificacao").alias("label"), col("features_conteudo").alias("features"))

# Spark 2.2.0
#data_to_test = MLUtils.convertVectorColumnsFrom(data_to_test, "features")

data = data_to_test.rdd.map(lambda row: LabeledPoint(row.label, Vectors.dense(((row.features).toArray()))))   
    
    # Divide data
splits = data.randomSplit([0.7, 0.3], 1234)
data_train = splits[0]
data_test = splits[1]
    
    # Spark 2.2.0
    #nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    
    # Spark 1.6.0
nb = NaiveBayes.train(data_train)
    
    # Spark 2.2.0
#    nv_m = nb.fit(data_train)
#    predictions = nv_m.transform(data_test)
#    evaluator_nv = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
#    accuracy = evaluator_nv.evaluate(predictions)
    
# Spark 1.6.0
predictions = data_test.map(lambda p: (nb.predict(p.features), p.label))
accuracy = 1.0 * predictions.filter(lambda (x, v): x == v).count() / data_test.count()

print("Test set accuracy = " + str(accuracy))
# 50%, 100000 features
# Test set accuracy = 0.630712549234
# 50%, 50000 features
#Test set accuracy = 0.620083674123

predictionAndLabels = data_test.map(lambda lp: (float(nb.predict(lp.features)), lp.label))

metrics = MulticlassMetrics(predictionAndLabels)   
print("Recall NB = %s" % metrics.recall())
print("Precision NB = %s" % metrics.precision())
print("F1 measure NB = %s" % metrics.fMeasure())
    
    ###################################################
    # Random Forest
    
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

data = data_to_test

# StringIndex
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# VectorIndexer
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=29).fit(data)

# Divide
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=5)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
   
#Test Error = 0.689862
    
    ###################################################
    # Logistic Regression
    
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics

# Load training data in LIBSVM format
data = data_to_test.rdd.map(lambda row: LabeledPoint(row.label, Vectors.dense(((row.features).toArray()))))   

# Split data into training (60%) and test (40%)
training, test = data.randomSplit([0.6, 0.4], seed=11L)
training.cache()

# Run training algorithm to build the model
model = LogisticRegressionWithLBFGS.train(training, numClasses=29)

# Compute raw scores on the test set
predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)
    
    #0.604911584393
    
    ###################################################
    # Multilayer perceptron classifier
    
    
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load training data
data = data_to_test

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]
# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [50000, 500, 10, 29]
# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234L)
# train the model
model = trainer.fit(train)
# compute precision on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="precision")



print("Precision:" + str(evaluator.evaluate(predictionAndLabels)))
# 0.553288779458
    sc.stop()
