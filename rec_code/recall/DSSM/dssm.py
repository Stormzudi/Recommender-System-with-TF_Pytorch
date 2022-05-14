# coding: utf-8
import os
import random
from datetime import datetime
from deepmatch.models import DSSM
from multiprocessing import Process
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from deepctr.feature_column import DenseFeat, SparseFeat, get_feature_names
from offline.feature_processing.v2 import (
    get_basic_user_feat,
    get_cate_stat_feat,
    get_hours,
    get_item_stat_feat,
    get_store_stat_feat,
    get_user_stat_feat,
)
from offline.rank.rank_model import RankModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from common.database import MysqlConn, redis_conn, spark
from common.log_utils import Loggers
from common.run_time import func_run_time_monitor
from setting.default import (
    ALS_CONFIG,
    DSSM_CONFIG,
    IS_DEV,
    ITEM_ACT_TYPES,
    RECALL_PARAMS,
    REDIS_EXPIRE_TIME,
    REDIS_STORE_FEATURES,
    REDIS_USER_FEATURES,
    SITE_ITEM_TYPES,
    MysqlConfig,
)

from .utils import insert_embedding, clear_exists_embs, get_raw_data, get_model_feature_names

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

schema = MysqlConfig['db']
logger = Loggers("DSSM_recall").logger


def get_train_eval_date(data, time_key="datetime"):

    days = list(sorted(set([str(e)[:10] for e in data[time_key].unique()])))

    train_start_date = days[1]
    train_end_date = days[int(0.8 * len(days))]
    eval_start_date = days[int(0.8 * len(days)) + 1]
    eval_end_date = days[-1]

    logger.info(f"""
    train_start_date: {train_start_date}
    train_end_date: {train_end_date}
    eval_start_date: {eval_start_date}
    eval_end_date: {eval_end_date}
    """)

    return train_start_date, train_end_date, eval_start_date, eval_end_date


class RecallTrain:
    def __init__(self, site_id, item_type, model_type, conn):
        self.site_id = site_id
        self.item_type = item_type
        self.model_type = model_type
        self.model_name = (
            self.site_id + "_" + self.item_type + "_" + model_type
        )
        self.conn = conn
        self.model_path = DSSM_CONFIG["model_path"]
        self.neg_sample_ratio = DSSM_CONFIG["neg_sample_ratio"]
        self.target = "label"

    # user行为时间特征
    def get_user_time_feat(self, data, start_date, end_date):
        label_end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1)
        label_set = data[
            (data["datetime"] >= end_date)
            & (data["datetime"] < label_end_date)
        ].reset_index(drop=True)
        label_user_index = (
            label_set[["user_id_index"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        actions = data[
            (data["datetime"] >= start_date) & (data["datetime"] < end_date)
        ]
        actions = (
            actions[
                actions["user_id_index"].isin(
                    label_user_index["user_id_index"].values
                )
            ]
            .sort_values(by="datetime", ascending=True)
            .reset_index(drop=True)
        )

        active_days = actions[["user_id_index", "datetime"]].reset_index(
            drop=True
        )
        active_days["datetime"] = (
            active_days["datetime"]
            .apply(lambda x: x.strftime("%Y-%m-%d"))
            .values
        )
        active_last = actions[["user_id_index", "datetime"]]
        active_first = actions[["user_id_index", "datetime"]]
        active_gap = actions.drop_duplicates(
            ["user_id_index", "datetime"], keep="first"
        )

        # 购买天数
        active_days = (
            active_days.groupby(["user_id_index", "datetime"])
            .size()
            .reset_index()
        )
        active_days = active_days.groupby("user_id_index").size().reset_index()
        active_days.rename(columns={0: "user_active_days"}, inplace=True)

        # end_date距离上一次购买时间(h)
        active_last = active_last.sort_values(by="datetime", ascending=False)
        active_last = active_last.drop_duplicates("user_id_index").reset_index(
            drop=True
        )
        active_last["user_active_last"] = active_last["datetime"].apply(
            lambda x: get_hours(x, end_date)
        )
        del active_last["datetime"]

        # end_date距离首次购买时间(h)
        active_first = active_first.sort_values(by="datetime", ascending=True)
        active_first = active_first.drop_duplicates(
            "user_id_index"
        ).reset_index(drop=True)
        active_first["user_active_first"] = active_first["datetime"].apply(
            lambda x: get_hours(x, end_date)
        )
        del active_first["datetime"]

        # 平均购买间隔_天
        active_gap = (
            active_gap.groupby("user_id_index")["datetime"]
            .apply(
                lambda x: np.diff(np.array(x)).mean()
                if np.diff(np.array(x)).mean()
                else 0
            )
            .reset_index()
        )
        active_gap["user_avg_buy_gap"] = active_gap["datetime"].dt.days
        del active_gap["datetime"]

        label_user_index = label_user_index.merge(
            active_days, on="user_id_index", how="left"
        )
        label_user_index = label_user_index.merge(
            active_last, on="user_id_index", how="left"
        )
        label_user_index = label_user_index.merge(
            active_first, on="user_id_index", how="left"
        )
        label_user_index = label_user_index.merge(
            active_gap, on="user_id_index", how="left"
        ).fillna(0)
        return label_user_index

    def get_feature_voc_size(self, data, sparse_features):
        feature_voc_size = {}
        for f in sparse_features:
            feature_voc_size[f] = data[f].max() + 1
        return feature_voc_size

    def get_feature_columns(
        self,
        feature_voc_size,
        user_sparse_features,
        user_dense_features,
        item_sparse_features,
        item_dense_features,
    ):

        user_feature_columns = [
            SparseFeat(
                feat, vocabulary_size=feature_voc_size[feat], embedding_dim=64
            )
            for i, feat in enumerate(user_sparse_features)
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in user_dense_features
        ]
        item_feature_columns = [
            SparseFeat(
                feat, vocabulary_size=feature_voc_size[feat], embedding_dim=64
            )
            for i, feat in enumerate(item_sparse_features)
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in item_dense_features
        ]
        feature_names = get_feature_names(
            user_feature_columns + item_feature_columns
        )
        return user_feature_columns, item_feature_columns, feature_names

    def train_eval_split(self, raw_data, user, item):
        """构造训练与评估数据集"""
        # import pdb;pdb.set_trace()
        item_to_cate = {}
        item_to_store = {}
        coupon_id_to_type = defaultdict(dict)
        for i in range(item.shape[0]):
            item_id_tmp = item.loc[i, "item_id_index"]
            cate_id_tmp = item.loc[i, "category1_id_index"]
            item_to_cate[item_id_tmp] = cate_id_tmp
            if self.item_type == "product":
                store_id_tmp = item.loc[i, "store_id_index"]
                item_to_store[item_id_tmp] = store_id_tmp
            elif self.item_type == "coupon":
                item_type_temp = item.loc[i, "item_type_index"]
                get_value_temp = item.loc[i, "get_value"]
                item_value_temp = item.loc[i, "item_value"]
                use_value_temp = item.loc[i, "use_value"]
                coupon_id_to_type[item_id_tmp]['item_type_index'] = item_type_temp
                coupon_id_to_type[item_id_tmp]['get_value'] = get_value_temp
                coupon_id_to_type[item_id_tmp]['item_value'] = item_value_temp
                coupon_id_to_type[item_id_tmp]['use_value'] = use_value_temp
        candidate_item = set(item["item_id_index"].unique())
        (
            self.train_start_date,
            self.train_end_date,
            self.eval_start_date,
            self.eval_end_date,
        ) = get_train_eval_date(raw_data)
        # 获取用户历史itemIndex
        user_history = raw_data.groupby("user_id_index")[
            "item_id_index"
        ].apply(set)
        user_history_dict = dict(zip(user_history.index, user_history.values))
        train_data = self.generate_dataset(
            raw_data,
            user,
            self.train_start_date,
            self.train_end_date,
            user_history_dict,
            item_to_cate,
            item_to_store,
            coupon_id_to_type,
            candidate_item,
        )
        eval_data = self.generate_dataset(
            raw_data,
            user,
            self.eval_start_date,
            self.eval_end_date,
            user_history_dict,
            item_to_cate,
            item_to_store,
            coupon_id_to_type,
            candidate_item,
        )
        # import pdb;pdb.set_trace()
        last_feature_data = pd.concat([eval_data, train_data]).reset_index(drop=True)
        last_user_feat = last_feature_data.drop_duplicates(
            "user_id_index", keep="last"
        )
        last_item_feat = last_feature_data.drop_duplicates(
            "item_id_index", keep="last"
        )
        train_data = shuffle(train_data)
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        eval_data = shuffle(eval_data)
        eval_data = eval_data.sample(frac=1).reset_index(drop=True)
        return train_data, eval_data, last_user_feat, last_item_feat

    def generate_dataset(
        self,
        data,
        user,
        train_start_date,
        train_end_date,
        user_history_dict,
        item_to_cate,
        item_to_store,
        coupon_id_to_type,
        candidate_item,
    ):
        """构造数据集"""
        dataset = pd.DataFrame()
        start_date = data["datetime"].min()

        for label_date in pd.date_range(
            start=pd.to_datetime(train_start_date),
            end=pd.to_datetime(train_end_date),
            freq="D",
        ).date:
            # import pdb;pdb.set_trace()
            label_date = pd.to_datetime(label_date)
            label_end_date = label_date + pd.DateOffset(days=1)
            label_set = data[
                (data["datetime"] >= label_date)
                & (data["datetime"] < label_end_date)
            ].reset_index(drop=True)
            if self.item_type == "product":
                label_index = label_set[
                    [
                        "user_id_index",
                        "item_id_index",
                        "category1_id_index",
                        "store_id_index",
                    ]
                ].drop_duplicates()
            elif self.item_type == "store":
                label_index = label_set[
                    ["user_id_index", "item_id_index", "category1_id_index"]
                ].drop_duplicates()
            elif self.item_type == 'coupon':
                label_index = label_set[
                    [
                        "user_id_index",
                        "item_id_index",
                        "category1_id_index",
                        "item_type_index",
                        "item_value",
                        "use_value",
                        "get_value",
                    ]
                ].drop_duplicates()
            history_user = data.loc[
                (data["datetime"] >= start_date)
                & (data["datetime"] < label_date),
                "user_id_index",
            ].values
            label_index = label_index[
                label_index["user_id_index"].isin(history_user)
            ].reset_index(drop=True)
            label_index[self.target] = 1
            label_index = self.neg_sampling(
                label_index,
                user_history_dict,
                item_to_cate,
                item_to_store,
                coupon_id_to_type,
                candidate_item,
            )

            user_feat = get_basic_user_feat(user)
            user_stat_feat = get_user_stat_feat(
                data, start_date, label_date, self.item_type
            )
            user_time_feat = self.get_user_time_feat(
                data, start_date, label_date
            )
            label_index = label_index.merge(
                user_feat, on="user_id_index", how="left"
            )
            # inner join 保留有历史行为的样本
            label_index = label_index.merge(
                user_stat_feat, on="user_id_index", how="inner"
            )
            label_index = label_index.merge(
                user_time_feat, on="user_id_index", how="left"
            )

            item_stat_feat = get_item_stat_feat(data, start_date, label_date)
            item_cate_stat_feat = get_cate_stat_feat(
                data, start_date, label_date, self.item_type
            )
            label_index = label_index.merge(
                item_stat_feat, on=["item_id_index"], how="left"
            )
            label_index = label_index.merge(
                item_cate_stat_feat, on=["category1_id_index"], how="left"
            )
            if self.item_type == "product":
                item_store_stat_feat = get_store_stat_feat(
                    data, start_date, label_date
                )
                label_index = label_index.merge(
                    item_store_stat_feat, on=["store_id_index"], how="left"
                )
            label_index = label_index.fillna(0)
            dataset = pd.concat([dataset, label_index]).reset_index(drop=True)
            logger.info(f"generate {label_date} data over")
        return dataset

    def neg_sampling(
        self,
        data,
        user_history_dict,
        item_to_cate,
        item_to_store,
        coupon_id_to_type,
        candidate_item,
    ):
        neg_sample_ratio = DSSM_CONFIG["neg_sample_ratio"]
        # 负采样
        neg_data_list = []
        for i in range(data.shape[0]):
            user_id_tmp = data.loc[i, "user_id_index"]
            candidate = candidate_item - user_history_dict[user_id_tmp] - {0}
            for j in range(neg_sample_ratio):
                item_id_neg = random.choice(list(candidate))
                category1_id_neg = item_to_cate[item_id_neg]
                if self.item_type == "product":
                    store_id_neg = item_to_store[item_id_neg]
                    row = {
                        "user_id_index": user_id_tmp,
                        "item_id_index": item_id_neg,
                        "category1_id_index": category1_id_neg,
                        "store_id_index": store_id_neg,
                        "label": 0,
                    }
                elif self.item_type == "store":
                    row = {
                        "user_id_index": user_id_tmp,
                        "item_id_index": item_id_neg,
                        "category1_id_index": category1_id_neg,
                        "label": 0,
                    }
                elif self.item_type == "coupon":
                    coupon_type_neg = coupon_id_to_type[item_id_neg]['item_type_index']
                    coupon_item_value_neg = coupon_id_to_type[item_id_neg]['item_value']
                    coupon_get_value_neg = coupon_id_to_type[item_id_neg]['get_value']
                    coupon_use_value_neg = coupon_id_to_type[item_id_neg]['use_value']
                    row = {
                        "user_id_index": user_id_tmp,
                        "item_id_index": item_id_neg,
                        "category1_id_index": category1_id_neg,
                        "item_type_index": coupon_type_neg,
                        "get_value": coupon_get_value_neg,
                        "item_value": coupon_item_value_neg,
                        "use_value": coupon_use_value_neg,
                        "label": 0,
                    }
                neg_data_list.append(row)

        data = data.append(neg_data_list, ignore_index=True)
        return data

    def eval_metric(
        self, model_version, train_sample_num, train_auc, eval_auc
    ):
        cur_dir_path = os.path.abspath(os.path.dirname(__file__))
        with open(cur_dir_path + "/eval_metric", "w") as f:
            f.write(
                "siteID,modelType,modelName,version,trainSampleNum,trainStartDate,trainEndDate,evalStartDate,"
                + "evalEndDate,properties\n"
            )
            line = [
                self.site_id,
                self.item_type,
                self.ctr_model_name,
                model_version,
                str(train_sample_num),
                self.train_start_date,
                self.train_end_date,
                self.eval_start_date,
                self.eval_end_date,
                "train_auc:"
                + str("%.3f" % train_auc)
                + ";eval_auc:"
                + str("%.3f" % eval_auc),
            ]
            f.write(",".join(line) + "\n")

    @func_run_time_monitor
    def train(self):
        """构造数据集, 训练排序模型"""
        user, item, raw_data, feature_dict = get_raw_data(
            self.site_id, self.item_type, self.conn
        )
        # import pdb;pdb.set_trace()
        id_index_to_id_dict = {}
        for key in ['user_id', 'item_id']:
            id_index_to_id_dict[key] = {}
            for _id, id_index in feature_dict[key].items():
                id_index_to_id_dict[key][id_index] = _id
        (
            self.user_sparse_features,
            self.user_dense_features,
            self.item_sparse_features,
            self.item_dense_features,
        ) = get_model_feature_names(self.item_type)
        self.all_spares_features = (
            self.user_sparse_features + self.item_sparse_features
        )
        self.all_dense_features = (
            self.user_dense_features + self.item_dense_features
        )
        feature_voc_size = self.get_feature_voc_size(
            raw_data, self.all_spares_features
        )
        (
            train_data,
            eval_data,
            last_user_feat,
            last_item_feat,
        ) = self.train_eval_split(raw_data, user, item)
        # Do simple Transformation for dense features
        train_data[self.all_dense_features] = np.log(
            train_data[self.all_dense_features] + 0.001
        )
        eval_data[self.all_dense_features] = np.log(
            eval_data[self.all_dense_features] + 0.001
        )
        scaler = MinMaxScaler()  # StandardScaler()
        scaler.fit(train_data[self.all_dense_features])
        train_data[self.all_dense_features] = scaler.transform(
            train_data[self.all_dense_features]
        )
        eval_data[self.all_dense_features] = scaler.transform(
            eval_data[self.all_dense_features]
        )

        (
            user_feature_columns,
            item_feature_columns,
            feature_names,
        ) = self.get_feature_columns(
            feature_voc_size, 
            self.user_sparse_features, self.user_dense_features, 
            self.item_sparse_features, self.item_dense_features
        )
        train_data_input = {name: train_data[name] for name in feature_names}
        eval_data_input = {name: eval_data[name] for name in feature_names}
        user_emb_input = {
            name: last_user_feat[name]
            for name in self.user_sparse_features + self.user_dense_features
        }
        item_emb_input = {
            name: last_item_feat[name]
            for name in self.item_sparse_features + self.item_dense_features
        }
        item_pos_id_dict = {}
        for i, item_id_index in enumerate(last_item_feat.item_id_index.values):
            item_id = id_index_to_id_dict['item_id'][item_id_index]
            item_pos_id_dict[i] = item_id
        user_pos_id_dict = {}
        for i, user_id_index in enumerate(last_user_feat.user_id_index.values):
            user_id = id_index_to_id_dict['user_id'][user_id_index]
            user_pos_id_dict[i] = user_id
        
        model_version = datetime.now().strftime("%Y%m%d%H%M")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        with tf.device("/cpu:0"):
            model = DSSM(user_feature_columns, item_feature_columns)
        try:
            parallel_model = multi_gpu_model(model, gpus=4)
            logger.info("Training using multiple GPUs..")
        except ValueError:
            parallel_model = model
            logger.info("Training using single GPU or CPU..")

        # Training
        EarlyStopping = tf.keras.callbacks.EarlyStopping
        ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
        Adam = tf.keras.optimizers.Adam
        callbacks = [
            EarlyStopping(monitor="val_auc", patience=1, mode="max"),
            ReduceLROnPlateau(
                monitor="auc", factor=0.3, patience=1, min_lr=1e-8, mode="max"
            ),
        ]
        parallel_model.compile(
            optimizer=Adam(lr=DSSM_CONFIG["learning_rate"]),
            loss="binary_crossentropy",
            metrics=[RankModel.auc, "binary_crossentropy"],
        )
        logger.info("Start Training")
        if eval_data.shape[0] > 0:
            history = parallel_model.fit(
                train_data_input,
                train_data[self.target].values,
                batch_size=DSSM_CONFIG["batch_size"],
                epochs=DSSM_CONFIG["epochs"],
                verbose=2,
                validation_data=(
                    eval_data_input,
                    eval_data[self.target].values,
                ),
                callbacks=callbacks,
            )
        else:
            history = parallel_model.fit(
                train_data_input,
                train_data[self.target].values,
                batch_size=DSSM_CONFIG["batch_size"],
                epochs=DSSM_CONFIG["epochs"],
                verbose=2,
                validation_split=0.2,
                callbacks=callbacks,
            )

        # Evaluation Record
        user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
        item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
        user_embs = user_embedding_model.predict(user_emb_input, batch_size=2 ** 12)
        item_embs = item_embedding_model.predict(item_emb_input, batch_size=2 ** 12)
        version_type = f"DSSM-{self.item_type}"
        logger.info(f"save embs to db with {version_type}")
        clear_exists_embs(
            self.conn, self.site_id, 
            f"{schema}.algo_user", version_type, 
            list(user_pos_id_dict.keys()), 'user_id',
        )
        clear_exists_embs(
            self.conn, self.site_id, 
            f"{schema}.algo_{self.item_type}", version_type,
            list(item_pos_id_dict.keys()), 'item_id',
        )
        insert_embedding(
            self.conn, self.site_id, f'{schema}.algo_user', version_type, 
            user_embs, user_pos_id_dict, feature_dict['user_id'], 'user_id'
        )
        insert_embedding(
            self.conn, self.site_id, f'{schema}.algo_{self.item_type}', version_type, 
            item_embs, item_pos_id_dict, feature_dict['item_id'], 'item_id'
        )
        logger.info(f"save embs to db over, total save: {len(item_embs) + len(user_embs)}")

        # import pdb;pdb.set_trace()
        train_sample_num = train_data.shape[0]
        train_auc = history.history["auc"][-1]
        eval_auc = history.history["val_auc"][-1]
        # self.eval_metric(model_version, train_sample_num, train_auc, eval_auc)


def process():
    conn = MysqlConn(**MysqlConfig)
    # RecallTrain('SCPG_huizhou_yxc', 'coupon', "DSSM", conn.conn).train()
    # RecallTrain('jdata_2019', 'store', "DSSM", conn.conn).train()
    from offline.rank.rank_main import get_scene_info

    scene_info = get_scene_info(conn.conn)
    for site_id, scene_detail in scene_info.items():
        for item_type in scene_detail["item_types"]:
            logger.info(f"Start Training {site_id} {item_type} DSSM model")
            recall_train = RecallTrain(site_id, item_type.lower(), "DSSM", conn.conn)
            p = Process(target=recall_train.train())
            p.start()
            p.join()
            logger.info("Finish Train")
