#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:49:33 2017

To use cluster it is necessary config hdfs with file
From: https://stackoverflow.com/questions/27299923/how-to-load-local-file-in-sc-textfile-instead-of-hdfs

Run command: pyspark --packages com.databricks:spark-csv_2.10:1.1.0 --executor-memory=1664M

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

#news_df = news_complete_data.sample(False, 0.10, 1234L)

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
news_df = news_df.where(news_df.classificacao != 'Producao')
news_df = news_df.where(news_df.classificacao != 'Mineracao')
news_df = news_df.where(news_df.classificacao != 'Especial')
news_df = news_df.where(news_df.classificacao != 'Credito')
news_df = news_df.where(news_df.classificacao != 'Comercio Exterior')
news_df = news_df.where(news_df.classificacao != 'Consumo')
news_df = news_df.where(news_df.classificacao != 'Petroleo')
news_df = news_df.where(news_df.classificacao != 'Turismo e lazer')
news_df = news_df.where(news_df.classificacao != 'Legislacao')
news_df = news_df.where(news_df.classificacao != 'Previdencia')
news_df = news_df.where(news_df.classificacao != 'Telecomunicacoes')
news_df = news_df.where(news_df.classificacao != 'Contas Publicas')
news_df = news_df.where(news_df.classificacao != 'Aviacao')
news_df = news_df.where(news_df.classificacao != 'Politica GDF')
news_df = news_df.where(news_df.classificacao != 'Habitacao')

#+--------------+------+
#| classificacao| count|
#+--------------+------+
#|       Cultura| 11146|
#|      Trabalho| 12736|
#|Infraestrutura| 13608|
#|       Energia| 13987|
#|    Transporte| 14276|
#|       Esporte| 15672|
#|   Agronegocio| 20053|
#| Meio Ambiente| 20939|
#|     Violencia| 29667|
#|      Educacao| 38337|
#|         Saude| 63747|
#|      Economia| 93441|
#|    Judiciario|114835|
#|      Politica|187803|
#+--------------+------+

news_df = news_df.where(news_df.classificacao != 'Cultura')
news_df = news_df.where(news_df.classificacao != 'Trabalho')
news_df = news_df.where(news_df.classificacao != 'Infraestrutura')
news_df = news_df.where(news_df.classificacao != 'Energia')
news_df = news_df.where(news_df.classificacao != 'Transporte')
news_df = news_df.where(news_df.classificacao != 'Agronegocio')
news_df = news_df.where(news_df.classificacao != 'Meio Ambiente')
news_df = news_df.where(news_df.classificacao != 'Violencia')
news_df = news_df.where(news_df.classificacao != 'Esporte')

count_news_df = news_df.groupBy('classificacao').count().sort(col("count").asc())

#customers.withColumn("movingAvg", avg(customers("amountSpent")).over(wSpec2))

#+-------------+------+
#|classificacao| count| avg
#+-------------+------+-------
#|     Educacao| 38337| 0,0769
#|        Saude| 63747| 0,1279
#|     Economia| 93441| 0,1875
#|   Judiciario|114835| 0,2305
#|     Politica|187803| 0,3769
#+-------------+------+

count_news_df.groupBy('classificacao').agg(stddev('Count'))

total_rows = news_df.count()
#Total 498163


# Total 100% = 709538
# Total 50% = 354452
# Total 40% = 283423
# Total 35% = 248081
# Total 30% = 212497
# Total 25% = 177110
# Total 20% = 141899
# Total 15% = 106560
# Total 10% = 70904



###################################################
# Get features from dataframe and transform in a vector  
newsGroup1 = news_df
data = newsGroup1

# StringIndex
str_idx_model = StringIndexer(inputCol="classificacao", outputCol="idx_classificacao").fit(data)
data_idx_clas = str_idx_model.transform(data)

#data_idx_clas.groupBy('idx_classificacao').count().sort(col("count").asc()).show()

#+-----------------+------+
#|idx_classificacao| count|
#+-----------------+------+
#|              4.0| 38337|
#|              3.0| 63747|
#|              2.0| 93441|
#|              1.0|114835|
#|              0.0|187803|
#+-----------------+------+

# Get stratified sample
newsSampled = data_idx_clas.sampleBy('idx_classificacao', fractions={0.0: .3769, 1.0: .2305, 2.0: .1875, 3.0: .1279, 4.0: .0769}, seed=1234L)
#newsSampled.groupBy('idx_classificacao').count().show()

#+-----------------+-----+
#|idx_classificacao|count|
#+-----------------+-----+
#|              1.0|26470|
#|              3.0| 8194|
#|              0.0|70589|
#|              4.0| 3030|
#|              2.0|17424|
#+-----------------+-----+


newsSample.stddev('').show()

data_idx_clas = newsSampled

# Tokenize
tk_model = Tokenizer(inputCol="conteudo", outputCol="tk_conteudo")
data_tk_cont = tk_model.transform(data_idx_clas)

#from pyspark.ml.feature import CountVectorizer
#data_vectorizer = CountVectorizer(inputCol="tk_conteudo", outputCol="vectorizer_conteudo").fit(data_tk_cont)
#len(data_vectorizer.vocabulary)
#262144

# N-gram
#from pyspark.ml.feature import NGram
#ngram = NGram(n=3, inputCol="tk_conteudo", outputCol="ngrams")
#data_ngram = ngram.transform(data_tk_cont)

# TF-IDF
hashingTF = HashingTF(inputCol="tk_conteudo", outputCol="v_conteudo", numFeatures=5000)
data_v_cont = hashingTF.transform(data_tk_cont)

idf = IDF(inputCol="v_conteudo", outputCol="features_conteudo")
idfModel = idf.fit(data_v_cont)
data_idf_cont = idfModel.transform(data_v_cont)

    ###################################################
    # K-means
    
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint  

data_to_test = data_idf_cont.select(col("features_conteudo").alias("features"))

parsedData = data_to_test.rdd.map(lambda line: (line.features).toArray())

#parsedData = data.map(lambda lp: (float(nb.predict(lp.features))))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 27, maxIterations=5,
        runs=2, initializationMode="random", epsilon=1e-4, seed=1234L)

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

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
splits = data.randomSplit([0.7, 0.3], 1234L)
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

#Test set accuracy = 0.716342195426

predictionAndLabels = data_test.map(lambda lp: (float(nb.predict(lp.features)), lp.label))

metrics = MulticlassMetrics(predictionAndLabels)

# Statistics by class
labels = data.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):    
    print("Class %s recall = %s" % (label, metrics.recall(label)))
    
#Class 0.0 recall = 0.680546396609
#Class 1.0 recall = 0.695679716204
#Class 2.0 recall = 0.804967979818
#Class 3.0 recall = 0.861762328213
#Class 4.0 recall = 0.834628190899

# Weighted stats
print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

#Weighted precision = 0.751056405489
#Weighted false positive rate = 0.123613762317
#Weighted false positive rate = 0.716342195426
    
    ###################################################
    # Random Forest
    
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.regression import LabeledPoint  

data = data_to_test

# StringIndex
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# VectorIndexer
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Divide
(trainingData, testData) = data.randomSplit([0.7, 0.3], 1234L)

# Train
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=6)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="recall")
metric = evaluator.evaluate(predictions)
print("recall = %g" % (metric))

#recall = 0.570034
#weightedPrecision = 0.58534
#Test Error = 0.429966

###############
# Cross Validation

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

data = data_to_test

# StringIndex
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# VectorIndexer
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Divide
(trainingData, testData) = data.randomSplit([0.7, 0.3], 1234L)

rf = RandomForestClassifier(labelCol="label", featuresCol="features")
evaluator = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName('accuracy') 

pipeline = Pipeline(stages=[rf])


paramGrid = ParamGridBuilder().addGrid(5000, [4, 5, 6]).build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=6)

model = crossval.fit(trainingData).bestModel
pr = model.transform(trainingData)
metric = evaluator.evaluate(pr)
print "Accuracy metric = %g" % metric

    ###################################################
    # Logistic Regression
    
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint  

data_to_test = data_idf_cont.select(col("idx_classificacao").alias("label"), col("features_conteudo").alias("features"))

# Load training data in LIBSVM format
data = data_to_test.rdd.map(lambda row: LabeledPoint(row.label, Vectors.dense(((row.features).toArray()))))   

# Split data into training (60%) and test (40%)
training, test = data.randomSplit([0.7, 0.4], 1234L)
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
    
#10%, 10000 features
#Precision = 0.604911584393
# 30%, 1000 features
#Precision = 0.570937364476
    
#Step Size: 1.000
#INFO optimize.LBFGS: Val and Grad Norm: 1.96684 (rel: 1.05e-05) 0.00462117

#10%, 5000
#Grad Norm: 1.35575 (rel: 3.98e-06) 0.00392328
#Precision = 0.593742730868
    
    ###################################################
    # Multilayer perceptron classifier
    
    
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

data_to_test = data_idf_cont.select(col("idx_classificacao").alias("label"), col("features_conteudo").alias("features"))

# Load training data
data = data_to_test

# Split the data into train and test
splits = data.randomSplit([0.7, 0.3], 1234L)
train = splits[0]
test = splits[1]
# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [5000, 500, 10, 29]
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
