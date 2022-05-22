#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo1.py
# @Author: Stormzudi
# @Date  : 2021/8/19 17:49



from gensim.models import KeyedVectors,word2vec,Word2Vec
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# you can modify your file path
file_train = '../../data/train_demo1.csv'


df_train = pd.read_csv(file_train)



# ==============appid split ===================
# 删除掉一些字符
dat = df_train['appid']
model = Word2Vec(dat, min_count=1, size= 50,workers=3, window =3, sg = 1)



a = lambda s: re.sub('[^A-Za-z0-9 ]+', ',', s)
data_str = map(a, df_train["appid"])
value = list(data_str)
vocab = list(set(str(value).split(' '))) # 转化成list 并去重

# 2. 建立词与id的映射关系
vocab2id = {}
for i, v in enumerate(vocab):
    vocab2id[v] = i + 2

values = np.array(vocab)




a = 1
