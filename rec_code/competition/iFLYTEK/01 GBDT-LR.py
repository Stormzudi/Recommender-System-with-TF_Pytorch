#!/usr/bin/env python
# coding: utf-8


# ## 1. 导入数据
# In[2]:


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import time
import datetime
# from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from scipy import sparse
from tqdm import tqdm_notebook
import re
## 存储文件
import pickle

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# 训练数据
df_train = pd.read_csv('data/train.csv')
# 测试数据
df_apply_new = pd.read_csv('data/apply_new.csv')
# 合并训练集，验证集
data = pd.concat([df_train,df_apply_new],axis=0,ignore_index=True)
data['label'] = data['label'].fillna(str(-1))


# # In[3]:
# data.head()
# # In[4]:
#
#
# data.shape # 数据维度
#
#
# # In[5]:
#
#
# data.columns  # 列信息
#
#
# # In[6]:
#
#
# data.isnull().sum() #检查空值
#
#
# # In[7]:
#
#
# df_apply_new.isnull().sum() #检查空值
#
#
# # ## 2. 特征工程-数据清洗、特征构建
#
# # ### 2.1 数据预处理
# # （1）统计**appid**中的个数并作为一个指标appid_num<br/>
# # （2）填补缺失值age(Null == 3)、gender(NaN == 3)<br/>
# # （3）province和city相加后成为一个新的指标p_c
#
# # In[12]:
#
#
# # 处理Age
# # 缺失值填充
# data['age'] = data['age'].fillna(0)
# data['age']
# a = data['age'].copy()
# # 统一字符类型转化成str()
# a = a.apply(lambda x: str(x).lower())
#
# def clean_data(string):
#     # 对数据清洗
#     string = re.sub(r"[^0-9()]", "", string)
#     return string.strip().lower()
# a = a.apply(lambda x: clean_data(x))
# data['age'] = a
# data['age']
#
#
# # In[13]:
#
#
# # 处理Gender
# # 缺失值填充
# data['gender'] = data['gender'].fillna(str(2))
# data['gender']
# g = data['gender'].copy()
# # 统一字符类型转化成str()
# g = g.apply(lambda x: str(x).lower())
#
# def clean_data(string):
#     # 对数据清洗
#     string = re.sub(r"[^0-9()]", "", string)
#     return string.strip().lower()
# g = g.apply(lambda x: clean_data(x))
# data['gender'] = g
# data['gender']
#
#
# # In[14]:
#
#
# # 处理appid
# appid_num = data['appid']
# def get_appid_num(string):
#     # 对数据清洗
#     string = string.split(',')
#     return len(string)
# appid_num = appid_num.apply(lambda x: get_appid_num(x))
# data['appid_num'] = appid_num
# data['appid_num']
#
#
# # In[15]:
#
#
# data
#
#
# # In[16]:
#
#
# data_pre = data[['id', 'label', 'gender', 'age', 'province','city', 'model', 'make', 'appid_num']]
# data_pre
#
#
# # In[17]:
#
#
# # labelencoder 转化
# encoder = ['province', 'city', 'model']
# lbl = LabelEncoder()
#
# for feat in encoder:
#     lbl.fit(data_pre[feat])
#     data_pre[feat] = lbl.transform(data_pre[feat])
# data_pre
#
#
# # In[18]:
#
#
# ## 存储文件
# import pickle
#
# ##存储中间特征矩阵便于再次访问
# with open('train_temp.pkl', 'wb') as file:
#     pickle.dump(data_pre, file)
#
#
# # ## 3. 训练模型
#
# # In[3]:


## 读取特征矩阵
with open('train_temp.pkl', 'rb') as file:
    data = pickle.load(file)
print('前10行的信息：\n', data.head(-10))

# In[4]:

"""
### 本次案例中：将所有的样本作为训练集。
### 使用全部的样本作为训练集，通过交叉验证的方法划分为：测试集+验证集（实质上没有使用测试集）
"""
total_train = data[data.label!= '-1']  # data.click != -1的样本为需要预测的样本集合
X_train = total_train
Y_train = total_train['label'] ##标签
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=5)
# print(x_train.shape, x_test.shape)

# In[5]:
print(x_train.shape, x_test.shape)
# In[6]:
print(y_train.shape, y_test.shape)

# In[7]:
# 这里只使用了ID特征 (特征：num_feature)，非ID特征之间是没有进行GBDT特征转化的过程
# num_feature = ['province', 'city', 'model']
num_feature = ['province', 'city', 'model']
pre_feature = ['gender', 'age' , 'appid_num']


# In[8]:
"""
### 用gbdt训练类别型变量，得到叶子节点拼接类别型，最后使用LR模型
"""
#用gbdt训练类别型变量，得到叶子节点拼接类别型，最后使用LR模型
# 模型部分

# Lightgbm参数学习的网站：https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
# n_estimators: 适合的提升树的数量
# num_leaves: 基学习器的最大树叶
lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=200,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=150, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)


# In[9]:

# 这里只使用了ID特征 (特征：num_feature)，非ID特征之间是没有进行GBDT特征转化的过程
train_csr = X_train[num_feature]
test_csr = x_test[num_feature]

# train_csr
# In[10]:

# 只提取最后100维数据，相当于embedding的维度是100维，相当于gbdt决策树
lgb_clf.fit(train_csr, Y_train.astype('int'))
new_feature_train = lgb_clf.predict(train_csr, pred_leaf = True)[:, -100:]
new_feature_test = lgb_clf.predict(test_csr, pred_leaf= True)[:, -100:]

# new_feature_train
# In[11]:

total_test = data[data.label == '-1']  # data.click != -1的样本为需要预测的样本集合
# total_test

# 得到待训练的10000个样本的数据
test_new = lgb_clf.predict(total_test[num_feature], pred_leaf = True)[:, -100:]
# test_new


# In[12]:
### 重命名GBDT的结果，在原始的X_train的特征中增加new_feature_train.shape[1]个决策树的特征
for i in range(new_feature_train.shape[1]):
    X_train['gbdt_'+str(i)] = new_feature_train[:, i]
    x_test['gbdt_'+str(i)] = new_feature_test[:, i]
    total_test['gbdt_'+str(i)] = test_new[:, i]

# In[13]:


### 拼接GBDT的结果的新的类别变量
### 这里将利用100维度向量作为特征
cate_feature = pre_feature + [i for i in X_train.columns if 'gbdt_' in i]
# In[14]:

# cate_feature
# In[15]:

# total_data

# In[16]:
### CTR预估常用方法，转换为One-hot高维稀疏数据，为了节省内存，使用CSR矩阵存储
total_data = pd.concat((X_train, total_test), axis = 0)
base_train_csr = sparse.csr_matrix((len(X_train), 0))
# base_test_csr = sparse.csr_matrix((len(x_test), 0))
base_test_csr = sparse.csr_matrix((len(total_test), 0)) # 测试用例

enc = OneHotEncoder()

"""
# 测试one-hot编码的过程：

total_data['adid'].values
# 用全部的样本取定义one-hot的语料
enc.fit(total_data['adid'].values.reshape(-1, 1))
# 针对训练集X_train得到该用户的one-hat向量
enc.transform(X_train['adid'].values.reshape(-1, 1))
# 转化成array()矩阵形式:
enc.transform(X_train['adid'].values.reshape(-1, 1)).toarray()

"""
for feature in cate_feature:
    enc.fit(total_data[feature].values.reshape(-1, 1))
    base_train_csr = sparse.hstack((base_train_csr, enc.transform(X_train[feature].values.reshape(-1, 1))), 'csr', 'bool')
#     base_test_csr = sparse.hstack((base_test_csr, enc.transform(x_test[feature].values.reshape(-1, 1))),'csr', 'bool')
    base_test_csr = sparse.hstack((base_test_csr, enc.transform(total_test[feature].values.reshape(-1, 1))),'csr', 'bool')
print('one-hot prepared !')


# In[17]:


print('训练集shape', base_train_csr.shape, '测试集shape', base_test_csr.shape)


# In[ ]:



"""
### LR模型  调参C

## 查看生成的one-hot()向量矩阵的形式：
base_train_csr.toarray()  
模型最后，可以用{'True', 'False'}输入到LR模型中做下游分析。

"""

# from sklearn.linear_model import LogisticRegression
# print('训练集shape', base_train_csr.shape, '测试集shape', base_test_csr.shape)
# # 使用验证集调参
# for c in [0.05, 0.1, 0.001, 0.01, 0.2, 0.005]:
#     print(c)
#     model = LogisticRegression(C=c, verbose=10)  # C = 5
#     model.fit(base_train_csr, Y_train.astype('int'))
#     train_pred = model.predict_proba(base_test_csr)[:, 1]
#     print('得到epcoh参数的过程loss', mean_squared_error(train_pred, y_test.array))
#     # print('得到epcoh参数的过程loss', log_loss(train_pred, y_test.array))
#     print('\n')

# 最后得到了使用的参数为0.2
# ## 4. 运用模型进行预测
# In[ ]:


# In[18]:
from sklearn.linear_model import LogisticRegression

c = 0.2
model = LogisticRegression(C=c, verbose=10)  # C = 5
model.fit(base_train_csr, Y_train.astype('int'))
train_pred = model.predict_proba(base_test_csr)[:, 1]

# train_pred
# In[19]:

prediction = model.predict(base_test_csr)
# prediction
# In[20]:

# 读入文件并写入预测值
label_submission = pd.read_csv('data/submit_sample.csv')
label_submission.head()

# In[21]:
label_submission['category_id']=prediction
# In[22]:
label_submission.to_csv("submission/submission_GBDT_LR_V3.csv", index=False)
# In[ ]:
