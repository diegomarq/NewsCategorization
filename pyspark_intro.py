#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:14:06 

From: https://spark.apache.org/docs/latest/ml-features.html
From: https://www.nodalpoint.com/spark-data-frames-from-csv-files-handling-headers-column-types/

@author: diego
"""
#Upload libriries
from pyspark.sql import SQLContext
from pyspark.sql.types import *

import unicodedata
import string

# Context
sqlContext = SQLContext(sc)

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
def remove_accents(word):
    return ''.join(x for x in unicodedata.normalize('NFKD', word) if x in string.ascii_letters).lower()


news_df = sqlContext.createDataFrame(newsNoFile, schema)