import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
from scipy import stats
import os
import platform
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import xgboost as xgb
from xgboost import plot_importance
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import getDataset
from model import TimeSeriesXgboost, TimeSeriesLightGBM
from utils import get_xgb_feat_importances, plotDataTrend


warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', 100)

### 1. load dataset
path = '.'
traindataset = getDataset(path)
traindataset["ts"] = traindataset["ts"].apply(lambda x: pd.to_datetime(x))
print(traindataset.shape)

### 2. train
# features
sparse_features = ['unit', 'year', 'month', 'day', 'week', 'geography_level_1', 'geography_level_2', 'geography_level_3',
                   'product_level_1', 'product_level_2', 'unit_pro', 'geo_pro']
dense_features = ['weight',
       'last14max', 'last14min', 'last14std', 'last14mean', 'last14median',
       'last14sum', 'last7max', 'last7min', 'last7std', 'last7mean',
       'last7median', 'last7sum', 'last3max', 'last3min', 'last3std',
       'last3mean', 'last3median', 'last3sum', 'last3value', 'last1value', 'last2mean',
       'last2sum', 'last2value', 'geo1mean14', 'geo2mean14', 'geo3mean14', 'pro1mean14',
       'pro2mean14', 'geo1median14', 'geo2median14', 'geo3median14',
       'pro1median14', 'pro2median14']

target = ['qty']


# 替换空值，和选择大于0的数据
traindataset = traindataset.dropna(subset=["qty"])
traindataset = traindataset[traindataset["qty"] >= 0]

# 归一化
traindataset['qty'] = np.log(traindataset['qty'] + 1)

# 训练集，验证集划分
traindataset = traindataset.dropna(axis=0, how='any')
train = traindataset[traindataset.ts <= pd.to_datetime("20210301")]
test = traindataset[traindataset.ts > pd.to_datetime("20210301")]

X_train, X_test, y_train, y_test = train[sparse_features + dense_features], test[sparse_features + dense_features], \
                                   train[target], test[target]

# training xgboost
bst = TimeSeriesXgboost(X_train, X_test, y_train, y_test)

# pre test_data
pre = bst.predict(X_test.values)
mae_norm = mean_absolute_error(y_test.values, pre)  # 归一化后的值
mae = mean_absolute_error(np.expm1(y_test.values), np.expm1(pre))

rmse = np.sqrt(mean_squared_error(np.expm1(y_test.values), np.expm1(pre)))

print("mae:",mae_norm)
print("mae:",mae)
print("rmse:",rmse)


# plot the feat importances
f, res = get_xgb_feat_importances(bst, sparse_features + dense_features)

plt.figure(figsize=(20, 10))
plt.barh(range(len(res)), res['Importance'][::-1], tick_label=res['Feature'][::-1])
plt.savefig('{}/output/xgb/xgboost_feature_importance.jpg'.format(path))
plt.show()


# plot data trend
data = X_test.copy()
data['qty'] = np.expm1(y_test)
data['pre_qty'] = np.expm1(pre)
data.groupby('unit').apply(plotDataTrend, path=path)