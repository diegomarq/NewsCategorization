#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:14:06 

From: https://spark.apache.org/docs/latest/ml-features.html
From: https://www.nodalpoint.com/spark-data-frames-from-csv-files-handling-headers-column-types/

Run with: PYSPARK_PYTHON=/home/diego/anaconda3/bin/python3.6 pyspark

@author: diego
"""

# Upoad File into data frame
news_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("delimiter", ";").load("/home/diego/Documents/Data/cod_clas_con_res_tit_semIne.csv")

news_df.printSchema()
#root
# |-- codigo: integer (nullable = true)
# |-- classificacao: string (nullable = true)
# |-- conteudo: string (nullable = true)
# |-- resumo: string (nullable = true)
# |-- titulo: string (nullable = true)

news_df.count()
# 714415

# Remove extra spaces
from pyspark.sql.functions import trim
news_df = news_df.withColumn("classificacao", trim(news_df.classificacao))
news_df = news_df.withColumn("conteudo", trim(news_df.conteudo))
news_df = news_df.withColumn("resumo", trim(news_df.resumo))
news_df = news_df.withColumn("titulo", trim(news_df.titulo))
	
# Clean everything inside <>
from pyspark.sql.functions import *    
news_df = news_df.withColumn('conteudo', regexp_replace('conteudo', '<.*?>', ''))

##############################
# Clean text removing accents
import unicodedata
from pyspark.sql.types import StringType

# Receives  a string as an argument
def remove_accents_(inputStr):
    # first, normalize strings:
    nfkdStr = unicodedata.normalize('NFKD', inputStr)
    # Keep chars that has no other char combined (i.e. accents chars)
    withOutAccents = u"".join([c for c in nfkdStr if not unicodedata.combining(c)])
    return withOutAccents

remove_accents = udf(lambda c: remove_accents_(c) if c != None else c, StringType())
    
##############################

news_df = news_df.withColumn('classificacao', remove_accents(news_df.classificacao))
news_df = news_df.withColumn('conteudo', remove_accents(news_df.conteudo))
news_df = news_df.withColumn('resumo', remove_accents(news_df.resumo))
news_df = news_df.withColumn('titulo', remove_accents(news_df.titulo))

# Group and count the categories present in the news
news_df.groupBy('classificacao').count().sort(col("count").desc()).show()

#+----------------+------+                                                       
#|   classificacao| count|
#+----------------+------+
#|        Politica|189087|
#|      Judiciario|115388|
#|        Economia| 93847|
#|           Saude| 63867|
#|        Educacao| 38408|
#|       Violencia| 29791|
#|   Meio Ambiente| 21106|
#|     Agronegocio| 20108|
#|         Esporte| 15714|
#|         Energia| 15210|
#|      Transporte| 14581|
#|  Infraestrutura| 13867|
#|        Trabalho| 12768|
#|         Cultura| 11170|
#|       Habitacao|  6217|
#|    Politica GDF|  6148|
#|         Aviacao|  6102|
#| Contas Publicas|  5431|
#|Telecomunicacoes|  5331|
#|      Legislacao|  5293|
#+----------------+------+
#only showing top 20 rows

##############################
# Tokenizer dataframe column
import numpy
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

tokenizer = Tokenizer(inputCol="resumo", outputCol="token_resumo")

countTokens = udf(lambda words: len(words), IntegerType())

tokenized = tokenizer.transform(news_df)
tokenized.select("resumo", "token_resumo").withColumn("num_token_resumo", 
                countTokens(col("token_resumo"))).show(truncate=False)




#newsFile = sc.textFile("file:///home/diego/Documents/Data/cod_clas_con_res_tit_semIne.csv")
#newsFile_map = newsFile.map(lambda l: l.split(';')).map(lambda l: (l[0], l[1], l[2], l[3], l[4]))

# Get fields from header
#header = newsFile.first()
#fields = [StructField(field_name, StringType(), True) for field_name in header.split(';')]
#schema = StructType(fields)

#news_df = spark.read.csv("/home/diego/Documents/Data/cod_clas_con_res_tit_semIne.csv",
#                         header=True, mode="DROPMALFORMED", schema=schema)

#sqlContext.registerDataFrameAsTable(news_df, "tmp_table_news")
#news_df.count()
# 714416
#sqlContext.sql('SELECT count(*), classificacao from  tmp_table_news group by classificacao order by count(*)').show()


# All the fields are stringType

# Create Schema from the fields


# Remove special caracters, transform to lower and remove accents



