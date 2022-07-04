import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import os
import platform
import xgboost as xgb
from xgboost import plot_importance
import lightgbm as lgb
from lightgbm import plot_importance


def TimeSeriesXgboost(X_train, X_test, y_train, y_test):
    ## 3.1 xgboost

    print('The shape of X_train:{}'.format(X_train.shape))
    print('The shape of X_test:{}'.format(X_test.shape))

    # params
    params = {
        'learning_rate': 0.2,
        'n_estimators': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'max_depth': 10,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'gamma': 0
    }

    print("Training xgboost ....")
    bst = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                           booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                           colsample_bytree=params['colsample_bytree'], random_state=0,
                           max_depth=params['max_depth'], gamma=params['gamma'],
                           min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    bst.fit(X_train.values, y_train.values)

    return bst


def TimeSeriesLightGBM(X_train, X_test, y_train, y_test):
    # 构造训练集
    dtrain = lgb.Dataset(X_train, y_train)
    dtest = lgb.Dataset(X_test, y_test)

    params = {
        'booster': 'gbtree',
        'objective': 'regression',
        'num_leaves': 31,
        'subsample': 0.8,
        'bagging_freq': 1,
        'feature_fraction ': 0.8,
        'slient': 1,
        'learning_rate ': 0.1,
        'seed': 0
    }

    num_rounds = 500

    # xgboost模型训练
    lgbmodel = lgb.train(params, dtrain, num_rounds, valid_sets=[dtrain, dtest],
                         verbose_eval=100, early_stopping_rounds=100)

    return lgbmodel
