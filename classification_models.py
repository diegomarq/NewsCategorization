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
news_df = news_df.where(news_df.resumo != '')

news_dfC2 = news_df.where(news_df.conteudo != '')
news_dfR1 = news_df.where(news_df.resumo != '')

# Total without resume
# 709664
news_df = news_dfC2
news_df = news_dfR1

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
#
#+-------------+------+
#|classificacao| count|avg
#+-------------+------+-----
#|     Educacao| 38343|
#|        Saude| 63751|
#|     Economia| 93450|
#|   Judiciario|114856|
#|     Politica|187862|
#+-------------+------+

news_df = news_df.where(news_df.classificacao != 'Educacao')
news_df = news_df.where(news_df.classificacao != 'Saude')
news_df = news_df.where(news_df.classificacao != 'Economia')
news_df = news_df.where(news_df.classificacao != 'Judiciario')
news_df = news_df.where(news_df.classificacao != 'Politica')

#count_news_df.withColumn('count', stddev('count'))

total_rows = news_df.count()
#Total 498163

#+-------------+------+
#|classificacao| count| avg
#+-------------+------+-------
#|     Educacao| 38337| 0,0769
#|        Saude| 63747| 0,1279
#|     Economia| 93441| 0,1875
#|   Judiciario|114835| 0,2305
#|     Politica|187803| 0,3769
#+-------------+------+

#Total 152084

#+--------------+-----+
#| classificacao|count| mean
#+--------------+-----+------
#|       Cultura|11146|0,0732
#|      Trabalho|12736|0,0837
#|Infraestrutura|13608|0,0894
#|       Energia|13987|0,0919
#|    Transporte|14276|0,0938
#|       Esporte|15672|0,1030
#|   Agronegocio|20053|0,1318
#| Meio Ambiente|20939|0,1376
#|     Violencia|29667|0,1950
#+--------------+-----+

#Total 58011

#+-----------------+-----+
#|    classificacao|count|
#+-----------------+-----+
#|         Especial| 1970|
#|          Credito| 2062|
#|Comercio Exterior| 2477|
#|          Consumo| 2926|
#|         Petroleo| 4237|
#|  Turismo e lazer| 4641|
#|       Legislacao| 5286|
#|      Previdencia| 5290|
#| Telecomunicacoes| 5329|
#|  Contas Publicas| 5419|
#|          Aviacao| 6082|
#|        Habitacao| 6146|
#|     Politica GDF| 6146|
#+-----------------+-----+

#---------------------------------------
# Resumo
#Total 498262

#+-------------+------+
#|classificacao| count|avg
#+-------------+------+-----
#|     Educacao| 38343|0,0769
#|        Saude| 63751|0,1279
#|     Economia| 93450|0,1875
#|   Judiciario|114856|0,2305
#|     Politica|187862|0,3770
#+-------------+------+

#Total 152102

#+--------------+-----+
#| classificacao|count|
#+--------------+-----+
#|       Cultura|11147|
#|      Trabalho|12737|
#|Infraestrutura|13611|
#|       Energia|13988|
#|    Transporte|14276|
#|       Esporte|15673|
#|   Agronegocio|20057|
#| Meio Ambiente|20945|
#|     Violencia|29668|
#+--------------+-----+

#Total 58020

#+-----------------+-----+
#|    classificacao|count|
#+-----------------+-----+
#|         Especial| 1970|
#|          Credito| 2062|
#|Comercio Exterior| 2479|
#|          Consumo| 2926|
#|         Petroleo| 4237|
#|  Turismo e lazer| 4641|
#|       Legislacao| 5288|
#|      Previdencia| 5291|
#| Telecomunicacoes| 5329|
#|  Contas Publicas| 5419|
#|          Aviacao| 6084|
#|     Politica GDF| 6146|
#|        Habitacao| 6148|
#+-----------------+-----+

###################################################
# Get features from dataframe and transform in a vector  
data = news_df

# StringIndex
str_idx_model = StringIndexer(inputCol="classificacao", outputCol="idx_classificacao").fit(data)
data_idx_clas = str_idx_model.transform(data)

#data_idx_clas.groupBy('idx_classificacao').count().sort(col("count").asc()).show()

#Group1
#+-----------------+------+
#|idx_classificacao| count|
#+-----------------+------+
#|              4.0| 38337|
#|              3.0| 63747|
#|              2.0| 93441|
#|              1.0|114835|
#|              0.0|187803|
#+-----------------+------+

#Group2

#+-----------------+-----+
#|idx_classificacao|count|
#+-----------------+-----+
#|              8.0|11146|
#|              7.0|12736|
#|              6.0|13608|
#|              5.0|13987|
#|              4.0|14276|
#|              3.0|15672|
#|              2.0|20053|
#|              1.0|20939|
#|              0.0|29667|
#+-----------------+-----+


# Get stratified sample
#Group1
newsSampled = data_idx_clas.sampleBy('idx_classificacao', fractions={0.0: .3769, 1.0: .2305, 2.0: .1875, 3.0: .1279, 4.0: .0769}, seed=1234L)

#Group2
newsSampled = data_idx_clas.sampleBy('idx_classificacao', fractions={
        0.0: .1950, 1.0: .1376, 2.0: .1318, 3.0: .1030, 4.0: .0938, 5.0: .0919, 6.0: .0894, 7.0: .0837, 8.0: .0732}, seed=1234L)
    
data_idx_clas = newsSampled

# Total 18649 Group 1 text
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

#-------------------------------------
# Resumo

#Total 125729 Group 1 resume
#
#+-----------------+-----+
#|idx_classificacao|count|
#+-----------------+-----+
#|              1.0|26407|
#|              3.0| 8056|
#|              0.0|70719|
#|              4.0| 2958|
#|              2.0|17589|
#+-----------------+-----+



# Tokenize
tk_model = Tokenizer(inputCol="resumo", outputCol="tk_conteudo")
data_tk_cont = tk_model.transform(data_idx_clas)

#from pyspark.ml.feature import CountVectorizer
#data_vectorizer = CountVectorizer(inputCol="tk_conteudo", outputCol="vectorizer_conteudo").fit(data_tk_cont)
#len(data_vectorizer.vocabulary)
#262144 text
#128426 resume

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
# Group1 5000
#Test set accuracy = 0.716342195426

#Group2 5000
#Test set accuracy = 0.771616829386

predictionAndLabels = data_test.map(lambda lp: (float(nb.predict(lp.features)), lp.label))

metrics = MulticlassMetrics(predictionAndLabels)

# Statistics by class
labels = data.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):    
    #print("Class %s recall = %s" % (label, metrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

#################################
#Group 1
#
#Class 0.0 F1 Measure = 0.714826650912
#Class 1.0 F1 Measure = 0.554560073513
#Class 2.0 F1 Measure = 0.632405063291
#Class 3.0 F1 Measure = 0.605682378059
#Class 4.0 F1 Measure = 0.431834929993

#################################
#Group2
#
#Class 0.0 F1 Measure = 0.847535132724
#Class 1.0 F1 Measure = 0.704680566022
#Class 2.0 F1 Measure = 0.830052162208
#Class 3.0 F1 Measure = 0.887329931973
#Class 4.0 F1 Measure = 0.712300447915
#Class 5.0 F1 Measure = 0.767953716514
#Class 6.0 F1 Measure = 0.597677826626
#Class 7.0 F1 Measure = 0.716055756324
#Class 8.0 F1 Measure = 0.796762069962

#################################
#Group3
#
#Class 0.0 F1 Measure = 0.705817782656
#Class 1.0 F1 Measure = 0.738675958188
#Class 2.0 F1 Measure = 0.821449616971
#Class 3.0 F1 Measure = 0.612220566319
#Class 4.0 F1 Measure = 0.804979253112
#Class 5.0 F1 Measure = 0.782465392221
#Class 6.0 F1 Measure = 0.548092084516
#Class 7.0 F1 Measure = 0.723880597015
#Class 8.0 F1 Measure = 0.770708283313
#Class 9.0 F1 Measure = 0.547281323877
#Class 10.0 F1 Measure = 0.728802588997
#Class 11.0 F1 Measure = 0.481298517996
#Class 12.0 F1 Measure = 0.383004926108

#################################
# Resumo
#Group1

#Class 0.0 F1 Measure = 0.691175265616
#Class 1.0 F1 Measure = 0.543088861661
#Class 2.0 F1 Measure = 0.597963540811
#Class 3.0 F1 Measure = 0.532303813426
#Class 4.0 F1 Measure = 0.390501319261

#################################
# Group2

#Class 0.0 F1 Measure = 0.72855972856
#Class 1.0 F1 Measure = 0.525715244567
#Class 2.0 F1 Measure = 0.646038365304
#Class 3.0 F1 Measure = 0.734880816072
#Class 4.0 F1 Measure = 0.549019607843
#Class 5.0 F1 Measure = 0.62236102436
#Class 6.0 F1 Measure = 0.427049456095
#Class 7.0 F1 Measure = 0.53531598513
#Class 8.0 F1 Measure = 0.572964034725

##################################
#Group3
#Class 0.0 F1 Measure = 0.528412419449
#Class 1.0 F1 Measure = 0.612750885478
#Class 2.0 F1 Measure = 0.660508083141
#Class 3.0 F1 Measure = 0.460485376478
#Class 4.0 F1 Measure = 0.605194805195
#Class 5.0 F1 Measure = 0.619032153297
#Class 6.0 F1 Measure = 0.416666666667
#Class 7.0 F1 Measure = 0.530669470733
#Class 8.0 F1 Measure = 0.594122319301
#Class 9.0 F1 Measure = 0.375366568915
#Class 10.0 F1 Measure = 0.465495608532
#Class 11.0 F1 Measure = 0.344594594595
#Class 12.0 F1 Measure = 0.245501285347


# Weighted stats
print("Weighted recall = %s" % metrics.weightedRecall)
print("Weighted precision = %s" % metrics.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
print("Weighted false positive rate = %s" % metrics.weightedTruePositiveRate)

#Group1
#Weighted recall = 0.645773020637
#Weighted precision = 0.692977161397
#Weighted F(1) Score = 0.656004813298
#Weighted F(0.5) Score = 0.674842676243
#Weighted false positive rate = 0.142221079593
#Weighted false positive rate = 0.645773020637


################################
#Group2

#Weighted recall = 0.771616829386
#Weighted precision = 0.775229941593
#Weighted F(1) Score = 0.772604843477
#Weighted F(0.5) Score = 0.773983721681
#Weighted false positive rate = 0.0301535837941
#Weighted true positive rate = 0.771616829386

##############################
#Group3
#Weighted recall = 0.687908309456
#Weighted precision = 0.707899192528
#Weighted F(1) Score = 0.694973240807
#Weighted F(0.5) Score = 0.702092207138
#Weighted false positive rate = 0.0256636782179
#Weighted true positive rate = 0.687908309456

#################################
# Resumo
#Group1
#Weighted recall = 0.616906570351
#Weighted precision = 0.672445289244
#Weighted F(1) Score = 0.629393909864
#Weighted F(0.5) Score = 0.651442257844
#Weighted false positive rate = 0.148148487969
#Weighted true positive rate = 0.616906570351

#################################
#Group2

#Weighted recall = 0.607052180004
#Weighted precision = 0.614180122241
#Weighted F(1) Score = 0.609314732612
#Weighted F(0.5) Score = 0.611930995376
#Weighted false positive rate = 0.0504114794602
#Weighted true positive rate = 0.607052180004

#################################
#Group3

#Weighted recall = 0.520600538651
#Weighted precision = 0.540277632187
#Weighted F(1) Score = 0.527929911377
#Weighted F(0.5) Score = 0.534756042945
#Weighted false positive rate = 0.0401837865892
#Weighted true positive rate = 0.520600538651


    ###################################################
    # Random Forest
    
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.regression import LabeledPoint  

data_to_test = data_idf_cont.select(col("idx_classificacao").alias("label"), col("features_conteudo").alias("features"))

data = data_to_test

# StringIndex
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# VectorIndexer
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(data)

# Divide
(trainingData, testData) = data.randomSplit([0.7, 0.3], 1234L)

# Train
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=5, impurity='entropy')

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
metric = evaluator.evaluate(predictions)
print("precision = %g" % (metric))

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="recall")
metric = evaluator.evaluate(predictions)
print("recall = %g" % (metric))

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
metric = evaluator.evaluate(predictions)
print("f1 = %g" % (metric))


#################################
#Group 1 5000 n= 5
#With 5 trees and impurity entropy

#precision = 0.561813
#recall = 0.561813
#f1 = 0.405899

##################################
#Group 2 5000 n = 9
#precision = 0.372329
#recall = 0.372329
#f1 = 0.312261

#################################
#Group 3 5000
#precision = 0.343658
#recall = 0.343658
#f1 = 0.310159

#################################
#Resumo
#Group 1 2000
#precision = 0.562485
#recall = 0.562485
#f1 = 0.405989

#Group2
#precision = 0.267641
#recall = 0.267641
#f1 = 0.199582

#Group3
#precision = 0.227163
#recall = 0.227163
#f1 = 0.211459


###############
# Cross Validation

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

data_to_test = data_idf_cont.select(col("idx_classificacao").alias("label"), col("features_conteudo").alias("features"))

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


paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [5, 6, 7]).build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=2)



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
model = LogisticRegressionWithLBFGS.train(training, numClasses=9)

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

# With not 
    
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
