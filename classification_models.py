#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:49:33 2017

To use cluster it is necessary config hdfs with file
From: https://stackoverflow.com/questions/27299923/how-to-load-local-file-in-sc-textfile-instead-of-hdfs

@author: diego
"""
from __future__ import print_function

from pyspark.sql import SQLContext
from pyspark.sql.types import *

import numpy as np
from pyspark.sql.functions import trim
import unicodedata
from pyspark.sql.types import StringType
from pyspark.sql.functions import isnan, when, count, col

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.functions import *    


if __name__ == "__main__":    
    
    sqlContext = SQLContext(sc)

    #news_complete_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', delimiter=';').load('file:///home/lara/diegomarques/cod_clas_con_res_tit_semIne.csv')
    news_complete_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', delimiter=';').load('hdfs://cluster0.local:8020/hdfs/diegomarques/data/cod_clas_con_res_tit_semIne.csv')

    
    # Spark 2.2.0 Upoad File into data frame
    news_complete_data = spark.read.format("csv").option("header", "true").option("inferSchema",
                               "true").option("delimiter", ";").load("cod_clas_con_res_tit_semIne.csv")  
    
    
    news_df = news_complete_data.sample(False, 0.05, 42)

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
    
    
    ###################################################
    # Get features from dataframe and transform in a vector  
    
    data = news_df

    # StringIndex
    str_idx_model = StringIndexer(inputCol="classificacao", outputCol="idx_classificacao").fit(data)
    data_idx_clas = str_idx_model.transform(data)
    
    # Tokenize
    tk_model = Tokenizer(inputCol="conteudo", outputCol="tk_conteudo")
    data_tk_cont = tk_model.transform(data_idx_clas)
    
    hashingTF = HashingTF(inputCol="tk_conteudo", outputCol="v_conteudo", numFeatures=10000)
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
    
    from pyspark.sql.functions import col
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.regression import LabeledPoint    
    
    from pyspark.mllib import linalg as mllib_linalg
    from pyspark.ml import linalg as ml_linalg
    
#    def parseLine(line):
#        parts = line.split(',')
#        label = float(parts[0])
#        features = Vectors.dense([float(x) for x in parts[1].split(' ')])
#        return LabeledPoint(label, features)
    
    data_to_test = data_idf_cont.select(col("idx_classificacao").alias("label"), col("features_conteudo").alias("features"))
    
    

    def as_old(v):
        if isinstance(v, ml_linalg.SparseVector):
            return mllib_linalg.SparseVector(v.size, v.indices, v.values)
        if isinstance(v, ml_linalg.DenseVector):
            return mllib_linalg.DenseVector(v.values)
        raise ValueError("Unsupported type {0}".format(type(v)))
    
    
    #data = data.map(parseLine)
    
    # Divide data
    splits = data.randomSplit([0.7, 0.3], 1234)
    data_train = splits[0]
    data_test = splits[1]
    
    # Spark 2.2.0
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    
    # Spark 1.6.0
    nb = NaiveBayes.train(data_train)
    
    # Spark 2.2.0
    nv_m = nb.fit(data_train)
    predictions = nv_m.transform(data_test)
    evaluator_nv = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_nv.evaluate(predictions)
    
    # Spark 1.6.0
    predictions = data_test.map(lambda p: (nb.predict(p.features), p.label))
    accuracy = 1.0 * predictions.filter(lambda (x, v): x == v).count() / data_test.count()
    
    print("Test set accuracy = " + str(accuracy))
    
    sc.stop()
