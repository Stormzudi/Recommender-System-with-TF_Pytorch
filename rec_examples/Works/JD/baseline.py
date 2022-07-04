# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries


class Unit:  # 一个单元
    def __init__(self, demand_true, inventory, weight):
        """
        :param demand_true: 取出某个unit下，历史所有的样本   shape: (, 9)
        :param inventory:  某个unit的库存信息 shape: (1, ), dt=2021-03-01
        :param weight:  样本权重
        """
        # 数据集
        self.weight = weight  # 单元权重
        self.demand_true = demand_true.reset_index(drop=True)  # 单元真实需求量
        self.inventory = inventory['qty'].values[0]  # 单元库存
        self.demand_pred = pd.DataFrame([])  # 单元预测需求量
        # 时间
        self.last_dt = pd.to_datetime("20210223")  # 预测开始
        self.start_dt = pd.to_datetime("20210302")  # 开始时间
        self.end_dt = pd.to_datetime("20210607")  # 结束时间
        self.date_list = pd.date_range(start=self.start_dt, end=self.end_dt)  # 时间序列
        self.lead_time = 14  # 补货在途时间

    def forecast(self):  # 预测

        data_series = TimeSeries.from_dataframe(self.demand_true[['ts', 'qty']], 'ts', 'qty')  # 将数据转换成TimeSeries格式
        scaler_data = Scaler()  # 数据标准化模型
        series_transformed = scaler_data.fit_transform(data_series)  # 数据标准化
        train_transformed, val_transformed = series_transformed.split_after(
            self.last_dt)  # 从2月23日划分训练集和测试集，用2月23日之前的做训练
        date_series = datetime_attribute_timeseries(series_transformed, attribute='day', one_hot=True)  # 按日转换为时间序列
        scaler_date = Scaler()  # 时间序列标准化模型
        covariates = scaler_date.fit_transform(date_series)  # 时间序列标准化
        train_date, val_date = covariates.split_after(self.last_dt)  # 从2月23日划分训练集和测试集

        model = TCNModel(  # 使用Deep TCN模型
            dropout=0.3,  # 每个卷积层的dropout率
            batch_size=10,  # 每次训练中使用的时间序列(输入和输出序列)的数量
            n_epochs=3,  # 训练模型的迭代次数
            optimizer_kwargs={'lr': 1e-2},  # 学习率
            random_state=1,  # 控制权重初始化的布尔值
            input_chunk_length=28,  # 输入层大小
            output_chunk_length=2,  # 输出层大小
            kernel_size=10,  # 卷积层内核大小
            num_filters=5,  # 过滤器数量
            likelihood=GaussianLikelihood())  # 使用高斯分布概率模型

        model.fit(series=train_transformed,  # 2月23日之前数据作训练集拟合
                  past_covariates=train_date,  # 2月23日之前时间序列作训练集拟合
                  verbose=True)  # 打印进度

        forecast_series = model.historical_forecasts(  # 预测3月8日之后的数据
            series=series_transformed,  # 输入数据序列
            past_covariates=covariates,  # 输入时间序列
            num_samples=100,  # 从概率模型中进行采样的数据
            start=self.last_dt,  # 从3月8日进行预测，因为补货决策第一次在3月8日(用到的预测数据起始在3月9日)
            forecast_horizon=self.lead_time,  # 连续预测14天,保证在决策和预测时不会用到未来的数据
            stride=1,  # 移动窗口,对下一个14天进行预测
            retrain=False,  # 每次预测无需重新训练
            verbose=True)  # 打印进度

        forecast = scaler_data.inverse_transform(forecast_series)  # 归一化后数据转换 还原
        self.demand_pred = forecast.quantile_df()  # 预测作为需求量
        self.demand_pred = self.demand_pred.reset_index()
        self.demand_pred.columns = ['ts', 'qty']
        # self.demand_pred['ts'] = self.demand_pred.index  # 时间序列作为ts列
        # self.demand_pred = self.demand_pred.reset_index(drop=True)  # 重置索引
        # self.demand_pred.rename(columns={'0': 'qty'}, inplace=True)  # 列改名
        self.demand_pred['qty'] = self.demand_pred['qty'].diff()  # 需求累积量做diff得需求量
        self.demand_true['ts'] = pd.to_datetime(self.demand_true['ts'], format='%Y-%m-%d')  # 时间转换
        self.demand_true['qty'] = self.demand_true['qty'].diff()  # 需求累积量做diff得需求量

    def MSS(self, value) -> float:  # 求最大子序列和
        maxSum = 0.0
        nowSum = 0.0
        for val in value:
            nowSum += val
            if (nowSum > maxSum):
                maxSum = nowSum
            elif nowSum < 0:
                nowSum = 0
        return maxSum

    def replenish(self):  # 补货

        intransit = pd.Series(index=self.date_list.tolist(), data=[0.0] * (len(self.date_list)))  # 补货到达记录
        arrival_sum = 0.0  # 累计补货到达
        qty_replenish = pd.Series(index=self.date_list.tolist(), data=[0.0] * (len(self.date_list)))  # 存放补货记录

        for date in self.date_list:
            # 对于每一天，进行库存更新
            demand_today = self.demand_true[self.demand_true['ts'] == date]['qty'].values[0]  # 当天消耗
            arrival_today = intransit.get(date, default=0.0)  # 当天到达
            self.inventory = max(self.inventory + arrival_today - demand_today, 0.0)  # 库存更新
            arrival_sum += arrival_today  # 累计补货到达

            if date.dayofweek == 0:  # 周一为补货决策日
                qty_intransit = sum(intransit) - arrival_sum  # 补货在途
                # 需求量计算
                demand_future_sum = self.demand_pred[(self.demand_pred['ts'] > date - self.lead_time * date.freq) & (
                            self.demand_pred['ts'] <= date + self.lead_time * date.freq)]["qty"]  # 未来14天需求量
                demand_history_plus = \
                self.demand_true[(self.demand_true['ts'] <= date) & (self.demand_true['qty'] > 0)]["qty"]  # 历史真实正需求量
                demand_history_minus = \
                self.demand_true[(self.demand_true['ts'] <= date) & (self.demand_true['qty'] < 0)]["qty"]  # 历史真实负需求量
                # 安全库存和再补货点
                safety_stock = 1.35 * (0.9 * demand_future_sum.std() + 0.1 * demand_history_plus.std()) * (
                            self.lead_time ** 0.5) + demand_history_plus.mean() - demand_history_minus.mean()
                # 安全库存 = 服务系数*(未来比重系数*未来标准差+历史比重系数*历史标准差)*(补货时长**0.5)+历史正均值-历史负均值
                demand_total = 0.55 * self.MSS(demand_future_sum) + 0.2 * np.sum(demand_future_sum)
                # 需求总和 = 调整系数*未来最大子序列和+调整系数*未来总和
                reorder_point = demand_total + safety_stock  # 再补货点=未来需求+安全库存
                # 补货策略
                if self.inventory + qty_intransit < reorder_point:  # 是否补货判别
                    replenish = reorder_point - (self.inventory + qty_intransit)  # 计算补货量
                    intransit.at[date + self.lead_time * date.freq] = replenish  # 添加补货在途
                    qty_replenish.at[date] = replenish  # 添加补货记录

        return qty_replenish


def dealDataSet():
    # 导入数据
    demand_train_B = pd.read_csv("./dataset/demand_train_A.csv")  # 虚拟资源使用量_训练数据
    demand_test_B = pd.read_csv("./dataset/demand_test_A.csv")  # 虚拟资源使用量_测试数据
    inventory_info_B = pd.read_csv("./dataset/inventory_info_A.csv")  # 库存数量
    weight_B = pd.read_csv("./dataset/weight_A.csv")  # 库存单元权重
    geo_topo = pd.read_csv("./dataset/geo_topo.csv")  # 地理层级
    product_topo = pd.read_csv("./dataset/product_topo.csv")  # 产品层级
    # 列变更
    demand = pd.concat([demand_train_B, demand_test_B])  # 数据合并
    demand.drop(demand.columns[[0, 4, 6]], axis=1, inplace=True)  # 删除无用列
    weight_B.drop(weight_B.columns[[0]], axis=1, inplace=True)  # 删除无用列
    demand.rename(columns={'geography': 'geography_level_3', 'product': 'product_level_2'}, inplace=True)  # 修改列名，便于做合并
    # 数据合并
    demand = pd.merge(demand, geo_topo, on='geography_level_3', how='left')  # 地理层级合并
    demand = pd.merge(demand, product_topo, on='product_level_2', how='left')  # 产品层级合并
    demand = pd.merge(demand, weight_B, on='unit', how='left')  # 库存单元权重信息

    return demand, inventory_info_B


def groupDataSet(demand, inventory):
    res = pd.DataFrame(columns=["unit", "ts", "qty"])  # 输出
    t = 0
    for unit in demand.groupby("unit"):  # 遍历unit_list
        t += 1
        print('Num:{}, Current Unit:{}'.format(t, unit[0]))

        replenish_unit = Unit(unit[1], inventory[inventory['unit'] == unit[0]], unit[1]['weight'].values[0])  # 创建单元
        replenish_unit.forecast()  # 预测
        res_unit = replenish_unit.replenish()  # 补货，并返回补货序列
        res_unit = pd.DataFrame({"unit": unit[0], "ts": res_unit.index, "qty": res_unit.values})  # 单元数据
        res = pd.concat([res, res_unit[res_unit["ts"].apply(lambda x: x.dayofweek == 0)]])  # 合并

    res.to_csv("./dataset/submit.csv")  # 输出结果
    print("The results have been output!")  # 运行结束


if __name__ == '__main__':
    demand, inventory = dealDataSet()  # 加载与处理数据
    groupDataSet(demand, inventory)  # 分单元预测与补货决策