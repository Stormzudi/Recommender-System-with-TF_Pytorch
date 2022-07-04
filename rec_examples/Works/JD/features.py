
import pandas as pd
import numpy as np
import datetime
import os
from sklearn.preprocessing import LabelEncoder
import pickle


# 1.1
def addMonthDay(trainalldate):
    # 添加 year, month, day 日期特征
    trainalldate['year'] = trainalldate['ts'].dt.year
    trainalldate['month'] = trainalldate['ts'].dt.month
    trainalldate['day'] = trainalldate['ts'].dt.day
    trainalldate['week'] = trainalldate['ts'].dt.weekday

    return trainalldate


# 1.2 处理geography_level、product_level多级别特征
def geoproFeat(trainalldate, geo_topo, product_topo, weight_A):

    trainalldate = trainalldate.drop(['geography_level', 'product_level'], axis=1)
    trainalldate = pd.merge(trainalldate, geo_topo, how='left', left_on='geography', right_on='geography_level_3')
    trainalldate = pd.merge(trainalldate, product_topo, how='left', left_on='product', right_on='product_level_2')
    trainalldate = trainalldate.drop(['geography', 'product'], axis=1)

    # labelEncoder
    encoder = ['geography_level_1', 'geography_level_2', 'geography_level_3', 'product_level_1', 'product_level_2']
    # add feature
    # unit_all = ['unit_geo', 'unit_pro', 'geo_pro']
    unit_all = ['unit_pro', 'geo_pro']

    # trainalldate["unit_geo"] = trainalldate.apply(lambda x: f"{x['unit']}_{x['geography_level_3']}", axis=1)
    trainalldate["unit_pro"] = trainalldate.apply(lambda x: f"{x['unit']}_{x['product_level_2']}", axis=1)
    trainalldate["geo_pro"] = trainalldate.apply(lambda x: f"{x['geography_level_3']}_{x['product_level_2']}", axis=1)

    lbl = LabelEncoder()
    for feat in encoder + unit_all:
        lbl.fit(trainalldate[feat])
        trainalldate[feat] = lbl.transform(trainalldate[feat])

    # add the weight of each units
    trainalldate = pd.merge(trainalldate, weight_A, left_on='unit', right_on='unit')
    trainalldate = trainalldate.drop(['Unnamed: 0'], axis=1)

    # unit to unit_id
    enc_unit = lbl.fit(trainalldate['unit'])
    trainalldate['unit'] = enc_unit.transform(trainalldate['unit'])

    # unit id -> 反编码
    # enc_unit.inverse_transform(trainalldate['unit'])

    return trainalldate


def qtyGetKvalue(data, k):
    '''
    k: the last k-ist value of data
    '''
    data = data.sort_values(
        by=['ts'], ascending=True).reset_index(drop=True)
    data = data.iloc[len(data) - k, -1] if len(data) >= k else np.NaN
    return data

def qtyNewFeature(df, ts=np.nan):
    newdataset = pd.DataFrame()

    timeline = pd.date_range(df.ts.min(), df.ts.max())
    for t in timeline:

        # today
        ts = df[df.ts == t]

        # last 14 day information ...
        rdd = df[(df.ts >= t - datetime.timedelta(14)) & (df.ts < t)]

        last14max_dict = rdd.groupby('unit')['qty'].max().to_dict()
        last14min_dict = rdd.groupby('unit')['qty'].min().to_dict()
        last14std_dict = rdd.groupby('unit')['qty'].std().to_dict()
        last14mean_dict = rdd.groupby('unit')['qty'].mean().to_dict()
        last14median_dict = rdd.groupby('unit')['qty'].median().to_dict()
        last14sum_dict = rdd.groupby('unit')['qty'].sum().to_dict()

        ts['last14max'] = ts['unit'].map(last14max_dict)
        ts['last14min'] = ts['unit'].map(last14min_dict)
        ts['last14std'] = ts['unit'].map(last14std_dict)
        ts['last14mean'] = ts['unit'].map(last14mean_dict)
        ts['last14median'] = ts['unit'].map(last14median_dict)
        ts['last14sum'] = ts['unit'].map(last14sum_dict)

        # last 7 day information ..
        rdd = df[(df.ts >= t - datetime.timedelta(7)) & (df.ts < t)]

        last7max_dict = rdd.groupby('unit')['qty'].max().to_dict()
        last7min_dict = rdd.groupby('unit')['qty'].min().to_dict()
        last7std_dict = rdd.groupby('unit')['qty'].std().to_dict()
        last7mean_dict = rdd.groupby('unit')['qty'].mean().to_dict()
        last7median_dict = rdd.groupby('unit')['qty'].median().to_dict()
        last7sum_dict = rdd.groupby('unit')['qty'].sum().to_dict()

        ts['last7max'] = ts['unit'].map(last7max_dict)
        ts['last7min'] = ts['unit'].map(last7min_dict)
        ts['last7std'] = ts['unit'].map(last7std_dict)
        ts['last7mean'] = ts['unit'].map(last7mean_dict)
        ts['last7median'] = ts['unit'].map(last7median_dict)
        ts['last7sum'] = ts['unit'].map(last7sum_dict)

        # last 3 day information ...
        rdd = df[(df.ts >= t - datetime.timedelta(3)) & (df.ts < t)]

        last3max_dict = rdd.groupby('unit')['qty'].max().to_dict()
        last3min_dict = rdd.groupby('unit')['qty'].min().to_dict()
        last3std_dict = rdd.groupby('unit')['qty'].std().to_dict()
        last3mean_dict = rdd.groupby('unit')['qty'].mean().to_dict()
        last3median_dict = rdd.groupby('unit')['qty'].median().to_dict()
        last3sum_dict = rdd.groupby('unit')['qty'].sum().to_dict()
        last3value_dict = rdd.groupby('unit')['ts', 'qty'].apply(qtyGetKvalue, k=3).to_dict()

        ts['last3max'] = ts['unit'].map(last3max_dict)
        ts['last3min'] = ts['unit'].map(last3min_dict)
        ts['last3std'] = ts['unit'].map(last3std_dict)
        ts['last3mean'] = ts['unit'].map(last3mean_dict)
        ts['last3median'] = ts['unit'].map(last3median_dict)
        ts['last3sum'] = ts['unit'].map(last3sum_dict)
        ts['last3value'] = ts['unit'].map(last3value_dict)

        # last 1、2 day information ..
        rdd = df[(df.ts >= t - datetime.timedelta(1)) & (df.ts < t)]
        last1value_dict = rdd.groupby('unit')['qty'].sum().to_dict()
        ts['last1value'] = ts['unit'].map(last1value_dict)

        rdd = df[(df.ts >= t - datetime.timedelta(2)) & (df.ts < t)]
        last2mean_dict = rdd.groupby('unit')['qty'].mean().to_dict()
        last2sum_dict = rdd.groupby('unit')['qty'].sum().to_dict()
        last2value_dict = rdd.groupby('unit')['ts', 'qty'].apply(qtyGetKvalue, k=2).to_dict()

        ts['last2mean'] = ts['unit'].map(last2mean_dict)
        ts['last2sum'] = ts['unit'].map(last2sum_dict)
        ts['last2value'] = ts['unit'].map(last2value_dict)

        newdataset = pd.concat([newdataset, ts])
        if t.month == 1 and t.day == 1:
            print(t)
    return newdataset


def geoproNewFeature(df):
    newdataset = pd.DataFrame()

    timeline = pd.date_range(df.ts.min(), df.ts.max())
    for t in timeline:
        ts = df[df.ts == t]
        rdd = df[(df.ts >= t - datetime.timedelta(14)) & (df.ts < t)]

        # grouby for calculate mean&median
        geo1mean14_dict = rdd.groupby('geography_level_1')['qty'].mean().to_dict()
        geo2mean14_dict = rdd.groupby('geography_level_2')['qty'].mean().to_dict()
        geo3mean14_dict = rdd.groupby('geography_level_3')['qty'].mean().to_dict()
        pro1mean14_dict = rdd.groupby('product_level_1')['qty'].mean().to_dict()
        pro2mean14_dict = rdd.groupby('product_level_2')['qty'].mean().to_dict()
        geo1median14_dict = rdd.groupby('geography_level_1')['qty'].median().to_dict()
        geo2median14_dict = rdd.groupby('geography_level_2')['qty'].median().to_dict()
        geo3median14_dict = rdd.groupby('geography_level_3')['qty'].median().to_dict()
        pro1median14_dict = rdd.groupby('product_level_1')['qty'].median().to_dict()
        pro2median14_dict = rdd.groupby('product_level_2')['qty'].median().to_dict()

        # map to df
        ts['geo1mean14'] = ts['geography_level_1'].map(geo1mean14_dict)
        ts['geo2mean14'] = ts['geography_level_2'].map(geo2mean14_dict)
        ts['geo3mean14'] = ts['geography_level_3'].map(geo3mean14_dict)
        ts['pro1mean14'] = ts['product_level_1'].map(pro1mean14_dict)
        ts['pro2mean14'] = ts['product_level_2'].map(pro2mean14_dict)

        ts['geo1median14'] = ts['geography_level_1'].map(geo1median14_dict)
        ts['geo2median14'] = ts['geography_level_2'].map(geo2median14_dict)
        ts['geo3median14'] = ts['geography_level_3'].map(geo3median14_dict)
        ts['pro1median14'] = ts['product_level_1'].map(pro1median14_dict)
        ts['pro2median14'] = ts['product_level_2'].map(pro2median14_dict)

        #         # grouby rdd for calculate mean&median and transform DataFrame
        #         ts['geo1mean14'] = rdd.groupby('geography_level_1')['qty'].transform('mean')
        #         ts['geo2mean14'] = rdd.groupby('geography_level_2')['qty'].transform('mean')
        #         ts['geo3mean14'] = rdd.groupby('geography_level_3')['qty'].transform('mean')
        #         ts['pro1mean14'] = rdd.groupby('product_level_1')['qty'].transform('mean')
        #         ts['pro2mean14'] = rdd.groupby('product_level_2')['qty'].transform('mean')
        #         ts['geo1median14'] = rdd.groupby('geography_level_1')['qty'].transform('median')
        #         ts['geo2median14'] = rdd.groupby('geography_level_2')['qty'].transform('median')
        #         ts['geo3median14'] = rdd.groupby('geography_level_3')['qty'].transform('median')
        #         ts['pro1median14'] = rdd.groupby('product_level_1')['qty'].transform('median')
        #         ts['pro2median14'] = rdd.groupby('product_level_2')['qty'].transform('median')

        newdataset = pd.concat([newdataset, ts])
        # print(t)
        if t.month == 1 and t.day == 1:
            print(t)
    return newdataset



def getDataset(path):
    # load traindataset
    dataset_path = '{}/output/traindataset.pkl'.format(path)
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as fo:  # 读取pkl文件数据
            traindataset = pickle.load(fo, encoding='bytes')
            print(traindataset.head())
            return traindataset
    else:
        # load dataset
        print('Generative training dataset ...')
        demand_train_A = pd.read_csv('{}/dataset/demand_train_A.csv'.format(path))
        demand_test_A = pd.read_csv('{}/dataset/demand_test_A.csv'.format(path))
        geo_topo = pd.read_csv('{}/dataset/geo_topo.csv'.format(path))
        inventory_info_A = pd.read_csv('{}/dataset/inventory_info_A.csv'.format(path))
        product_topo = pd.read_csv('{}/dataset/product_topo.csv'.format(path))
        weight_A = pd.read_csv('{}/dataset/weight_A.csv'.format(path))

        demand_train_A["ts"] = demand_train_A["ts"].apply(lambda x: pd.to_datetime(x))
        demand_train_A.drop(['Unnamed: 0'], axis=1, inplace=True)
        demand_test_A["ts"] = demand_test_A["ts"].apply(lambda x: pd.to_datetime(x))
        demand_test_A.drop(['Unnamed: 0'], axis=1, inplace=True)
        dataset = pd.concat([demand_train_A, demand_test_A])

        # 删除掉出现 qty中出现负值的样本，这部分样本数值不对
        dataset = dataset[~(dataset.qty < 0)]
        first_dt = pd.to_datetime("20180604")
        all_date = (dataset.ts.max() - dataset.ts.min()).days + 1

        # 1.1 日期处理 （天数补齐）
        trainalldate = pd.DataFrame()
        for unit in dataset.unit.drop_duplicates():
            tmppd = pd.DataFrame(index=pd.date_range(first_dt, periods=all_date))
            tmppd['unit'] = unit
            tmppd = tmppd.reset_index()
            tmppd.columns = ['ts', 'unit']
            tmppd = pd.merge(left=tmppd, right=dataset[dataset.unit == unit], how='left', on=['ts', 'unit']
                             )
            trainalldate = pd.concat([trainalldate, tmppd])

        trainalldate = addMonthDay(trainalldate)

        # 1.2 处理geography_level、product_level多级别特征
        trainalldate = geoproFeat(trainalldate, geo_topo, product_topo, weight_A)

        # 1.3 使用qty滑窗添加新特征：last14max、last14min、last14std、last14mean、last14median、last14sum ...
        print("Generative time series features ....")
        traindataset = qtyNewFeature(trainalldate)

        # 1.4 过去14天下，在同geography与product的qty 资源使用量。
        print("Generative geo & pro features ....")
        traindatasetall = geoproNewFeature(traindataset)

        # save to pkl
        with open(dataset_path, 'wb') as fo:
            pickle.dump(traindatasetall, fo)

        return trainalldate


if __name__ == '__main__':
    getDataset(path='.')


