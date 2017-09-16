#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:14:06 2017

@author: diego
"""
#Upload libriries
from pyspark.sql import SQLContext
from pyspark.sql.types import *

# Context
sqlContext = SQLContext(sc)

# Upoad File
newsFile = sc.textFile("file:///home/diego/Documents/Data/cod_clas_con_res_tit_semIne.csv")

newsFile.count()
# 714416

header = newsFile.first()
# u'codigo;classificacao;conteudo;resumo;titulo'

fields = [StructField(field_name, StringType(), True) for field_name in header.split(';')]
# All the fields are stringType

# Rename field 'codigo'
fields[0].name = '_codigo'

# Create Schema from the fields
schema = StructType(fields)

# Get line with '_codigo'
newsHeader = newsFile.filter(lambda l: "_codigo" in l)
newsHeader.count() # see if there is more one line except the reader
newsHeader.collect() # see what is inside
newsNoHeader = newsFile.subtract(newsHeader) # get the body of rdd

# Create dataframe from rdd csv uploaded
newsNoHeader.count() # 714416 :|
news_df = sqlContext.createDataFrame(newsNoFile, schema)
# see the problems....


