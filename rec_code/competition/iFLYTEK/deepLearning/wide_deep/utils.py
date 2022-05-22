# 数据预处理代码

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import re


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_criteo_dataset(file_train, file_test, embed_dim=8, read_part=False, sample_num=100000, test_size=0.2):

    # 训练数据
    df_train = pd.read_csv(file_train)
    # 测试数据
    df_apply_new = pd.read_csv(file_test)
    # 合并训练集，验证集
    data_df = pd.concat([df_train, df_apply_new], axis=0, ignore_index=True)
    data_df['label'] = data_df['label'].fillna(0)

    len_train = len(df_train)
    len_apply_new = len(df_apply_new)

    # 增加六个维度
    data_df['province_sum'] = df_train.groupby(['province'])['label'].transform('sum')
    val = data_df['province'].value_counts()
    data_df['province_pre'] = data_df['province_sum'] / data_df['province'].apply(lambda x: val[x])

    data_df['city_sum'] = df_train.groupby(['city'])['label'].transform('sum')
    val = data_df['city'].value_counts()
    data_df['city_pre'] = data_df['city_sum'] / data_df['city'].apply(lambda x: val[x])

    data_df['model_sum'] = df_train.groupby(['model'])['label'].transform('sum')
    val = data_df['model'].value_counts()
    data_df['model_pre'] = data_df['model_sum'] / data_df['model'].apply(lambda x: val[x])


    # 稀疏特征 label 类型的特征 不需要归一化，非连续
    sparse_features = ['gender', 'province', 'city', 'make', 'model']
    # 稠密特征 需要归一化 连续性的特征
    dense_features = ['age']


    def clean_data(string):
        # 对数据清洗
        string = re.sub(r"[^0-9()]", "", string)
        return string.strip().lower()


    # ==============Age ===================
    # 处理Age
    # 缺失值填充
    data_df['age'] = data_df['age'].fillna(-1)
    a = data_df['age'].copy()
    a = a.apply(lambda x: str(x).lower())
    # 统一字符类型转化成str()
    a = a.apply(lambda x: clean_data(x))
    data_df['age'] = a
    data_df['age'] = data_df['age'].astype('int')  # 转换数据类型为 int 类型


    # ==============appid_num ===================
    appid_num = data_df['appid']
    def get_appid_num(string):
        # 对数据清洗
        string = string.split(',')
        return len(string)

    appid_num = appid_num.apply(lambda x: get_appid_num(x))
    data_df['appid_num'] = appid_num
    dense_features = dense_features + ['appid_num']


    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # ==============gender ===================
    data_df['gender'] = data_df['gender'].astype('str')  # 转换数据类型为 int 类型


    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    #
    # # ==============appid split ===================
    # # 删除掉一些字符
    # a = lambda s: re.sub('[^A-Za-z0-9 ]+', ' ', s)
    # data_str = map(a, data_df["appid"])
    # value = list(data_str)
    # vocab = list(set(str(value).split(' '))) # 转化成list 并去重
    #
    # # 2. 建立词与id的映射关系
    # vocab2id = {}
    # for i, v in enumerate(vocab):
    #     vocab2id[v] = i + 2
    #
    # values = np.array(vocab)
    #
    # onehot_encoder = OneHotEncoder(sparse=False)
    # values = values.reshape(len(vocab), 1)
    # onehot_encoded = onehot_encoder.fit_transform(values)
    #
    # df_sparse_features = data_df[sparse_features].str.split(' ', expand=True)
    # data_df = data_df[dense_features]





    # ====================================================
    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])


    # 统计dense_features、sparse_features每个特征的个数和
    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
                        for feat in sparse_features]]

    # 划分训练集和测试集
    """
    ### 本次案例中：将所有的样本作为训练集。
    ### 使用全部的样本作为训练集，通过交叉验证的方法划分为：测试集+验证集
    """
    data_df_1 = data_df[data_df.label != '-1']  # data.click != -1的样本为训练的样本集合
    train, test = train_test_split(data_df_1, test_size=test_size)

    train_X = [train[dense_features].values.astype('int32'), train[sparse_features].values.astype('int32')]
    train_y = train['label'].values.astype('int32')
    test_X = [test[dense_features].values.astype('int32'), test[sparse_features].values.astype('int32')]
    test_y = test['label'].values.astype('int32')


    # 划分需要预测的样本集
    data_df_2 = data_df[data_df.label == '-1']  # data.click == -1的样本为需要预测的样本集合
    total_test = [data_df_2[dense_features].values.astype('int32'), train[sparse_features].values.astype('int32')]

    return feature_columns, (train_X, train_y), (test_X, test_y), total_test





if __name__ == '__main__':
    read_part = True
    sample_num = 6000000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 256
    epochs = 30

    # you can modify your file path
    file_train = '../../data/train.csv'
    file_test = '../../data/apply_new.csv'

    # ========================== Create dataset =======================
    feature_columns, train, test, vail = create_criteo_dataset(file_train=file_train,
                                                               file_test=file_test,
                                                               embed_dim=embed_dim,
                                                               read_part=read_part,
                                                               sample_num=sample_num,
                                                               test_size=test_size)

    a = 1





