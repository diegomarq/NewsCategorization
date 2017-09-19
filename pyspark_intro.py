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

#news_df.printSchema()
#root
# |-- codigo: integer (nullable = true)
# |-- classificacao: string (nullable = true)
# |-- conteudo: string (nullable = true)
# |-- resumo: string (nullable = true)
# |-- titulo: string (nullable = true)

#news_df.count()
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

# Null categories
from pyspark.sql.functions import isnan, when, count, col

news_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in news_df.columns]).show()
#
#+------+-------------+--------+------+------+                                   
#|codigo|classificacao|conteudo|resumo|titulo|
#+------+-------------+--------+------+------+
#|     0|            0|     127|  4753|     3|
#+------+-------------+--------+------+------+

# column conteudo with empty value
news_df[news_df.conteudo == ''].count()
#4750

# Get only conteudo not empty
news_df = news_df.where(news_df.conteudo != '')

# Now    
#
#+------+-------------+--------+------+------+                                   
#|codigo|classificacao|conteudo|resumo|titulo|
#+------+-------------+--------+------+------+
#|     0|            0|       0|     2|     3|
#+------+-------------+--------+------+------+


##############################

news_df = news_df.withColumn('classificacao', remove_accents(news_df.classificacao))
news_df = news_df.withColumn('conteudo', remove_accents(news_df.conteudo))
news_df = news_df.withColumn('resumo', remove_accents(news_df.resumo))
news_df = news_df.withColumn('titulo', remove_accents(news_df.titulo))

# Group and count the categories present in the news
news_df.groupBy('classificacao').count().sort(col("count").desc()).show()
#
#+----------------+------+                                                       
#|   classificacao| count|
#+----------------+------+
#|        Politica|187803|
#|      Judiciario|114835|
#|        Economia| 93441|
#|           Saude| 63747|
#|        Educacao| 38337|
#|       Violencia| 29667|
#|   Meio Ambiente| 20939|
#|     Agronegocio| 20053|
#|         Esporte| 15672|
#|      Transporte| 14276|
#|         Energia| 13987|
#|  Infraestrutura| 13608|
#|        Trabalho| 12736|
#|         Cultura| 11146|
#|    Politica GDF|  6146|
#|       Habitacao|  6146|
#|         Aviacao|  6082|
#| Contas Publicas|  5419|
#|Telecomunicacoes|  5329|
#|     Previdencia|  5290|
#+----------------+------+

#only showing top 20 rows

news_df.groupBy('classificacao').count().sort(col("count").asc()).show()
#
#+-----------------+-----+                                                       
#|    classificacao|count|
#+-----------------+-----+
#|         Producao|  529|
#|        Mineracao|  751|
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
#|     Politica GDF| 6146|
#|        Habitacao| 6146|
#|          Cultura|11146|
#|         Trabalho|12736|
#|   Infraestrutura|13608|
#|          Energia|13987|
#|       Transporte|14276|
#+-----------------+-----+
#only showing top 20 rows


cla_count_asc_df = news_df.groupBy('classificacao').count().sort(col("count").asc())

# Number of classifcations or categories
cla_count_asc_df.count()
#29                          



##############################
# Tokenizer dataframe column
#import numpy
#from pyspark.ml.feature import Tokenizer, RegexTokenizer
#from pyspark.sql.functions import col, udf
#from pyspark.sql.types import IntegerType
#
#tokenizer = Tokenizer(inputCol="resumo", outputCol="token_resumo")
#
#countTokens = udf(lambda words: len(words), IntegerType())
#
#tokenized = tokenizer.transform(news_df)
#tokenized.select("resumo", "token_resumo").withColumn("num_token_resumo", 
#                countTokens(col("token_resumo"))).show(truncate=False)




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



