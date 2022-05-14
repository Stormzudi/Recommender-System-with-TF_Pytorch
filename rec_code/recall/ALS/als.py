# coding: utf-8
"""
ALS recall
"""
import json
import os
import gc
import time

import pandas as pd
from sklearn.metrics import pairwise
from sklearn.preprocessing import LabelEncoder

from pyspark import Row
from pyspark.sql.functions import col, lower
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from offline.recall.base import hot_recall, new_recall
from common.database import MysqlConn, redis_conn, spark
from common.log_utils import Loggers
from common.run_time import func_run_time_monitor
from setting.default import (
    IS_DEV,
    SITE_ITEM_TYPES,
    ITEM_ACT_TYPES,
    REDIS_STORE_FEATURES,
    REDIS_USER_FEATURES,
    MysqlConfig,
    ALS_CONFIG,
    RECALL_PARAMS,
    REDIS_EXPIRE_TIME,
)

from offline.recall.utils import recall_n_calculate, get_act_data, get_item_data
from .utils import get_expose_item_id, get_rating_by_df

schema = MysqlConfig["db"]

logger = Loggers("recall_processer_v2").logger


class AlsRecommender:
    """
    This a collaborative filtering recommender with Alternating Least Square
    Matrix Factorization, which is implemented by Spark
    """

    def __init__(self, spark_session, item_df, rating_df):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.item_df = item_df
        self.rating_df = rating_df
        self.model = ALS(
            userCol="user_id",
            itemCol="item_id",
            ratingCol="rating",
            coldStartStrategy="drop",
        )

    def tune_model(self, split_ratio=(6, 2, 2)):
        """
        Hyperparameter tuning for ALS model
        Parameters
        ----------
        maxIter: int, max number of learning iterations
        regParams: list of float, regularization parameter
        ranks: list of float, number of latent factors
        split_ratio: tuple, (train, validation, test)
        """
        # split data
        train, val, test = self.rating_df.randomSplit(split_ratio)
        # holdout tuning
        self.model = tune_ALS(
            self.model, train, val, self.max_iter, self.reg_params, self.rank
        )
        # test model
        predictions = self.model.transform(test)
        evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        logger.info("The out-of-sample RMSE of the best tuned model is:", rmse)
        # clean up
        del train, val, test, predictions, evaluator
        gc.collect()

    def set_model_params(self, maxIter, regParam, rank):
        """
        set model params for pyspark.ml.recommendation.ALS
        Parameters
        ----------
        maxIter: int, max number of learning iterations
        regParams: float, regularization parameter
        ranks: float, number of latent factors
        """
        self.max_iter = maxIter
        self.reg_param = regParam
        self.rank = rank
        self.model = (
            self.model.setMaxIter(maxIter).setRank(rank).setRegParam(regParam)
        )
        self.model = self.model.fit(self.rating_df)

    def _regex_matching(self, fav_item):
        """
        return the closest matches via SQL regex.
        If no match found, return None
        Parameters
        ----------
        fav_item: str, name of user input movie
        Return
        ------
        list of indices of the matching movies
        """
        logger.info("input item_id:", fav_item)
        matchesDF = self.item_df.filter(
            lower(col("item_name")).like("%{}%".format(fav_item.lower()))
        ).select("item_id", "item_name")
        if not len(matchesDF.take(1)):
            logger.info("No match is found")
        else:
            item_ids = matchesDF.rdd.map(lambda r: r[0]).collect()
            item_names = matchesDF.rdd.map(lambda r: r[1]).collect()
            logger.info(
                "Found possible matches in our database: "
                "{0}\n".format([x for x in item_names])
            )
            return item_ids

    def _append_ratings(self, user_id, item_ids):
        """
        append a user's item ratings to ratingsDF
        Parameter
        ---------
        user_id: int, user_id of a user
        item_ids: int, item_ids of user's favorite items
        """
        # create new user rdd
        user_rdd = self.sc.parallelize(
            [(user_id, item_id, 5.0) for item_id in item_ids]
        )
        # transform to user rows
        user_rows = user_rdd.map(
            lambda x: Row(
                user_id=int(x[0]), item_id=int(x[1]), rating=float(x[2])
            )
        )
        # transform rows to spark DF
        userDF = self.spark.createDataFrame(user_rows).select(
            self.rating_df.columns
        )
        # append to ratingsDF
        self.rating_df = self.rating_df.union(userDF)

    def _create_inference_data(self, user_id, item_ids):
        """
        create a user with all items except ones were rated for inferencing
        """
        # filter items
        other_item_ids = (
            self.item_df.filter(~col("item_id").isin(item_ids))
            .select(["item_id"])
            .rdd.map(lambda r: r[0])
            .collect()
        )
        # create inference rdd
        inferenceRDD = self.sc.parallelize(
            [(user_id, item_id) for item_id in other_item_ids]
        ).map(
            lambda x: Row(
                user_id=int(x[0]),
                item_id=int(x[1]),
            )
        )
        # transform to inference DF
        inferenceDF = self.spark.createDataFrame(inferenceRDD).select(
            ["user_id", "item_id"]
        )
        return inferenceDF

    def _inference(self, model, fav_item, n_recommendations):
        """
        return top n movie recommendations based on user's input movie
        Parameters
        ----------
        model: spark ALS model
        fav_item: str, name of user input movie
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar movie recommendations
        """
        # create a user_id
        user_id = self.rating_df.agg({"user_id": "max"}).collect()[0][0] + 1
        # get item_ids of favorite
        item_ids = self._regex_matching(fav_item)
        # append new user with his/her ratings into data
        self._append_ratings(user_id, item_ids)
        # matrix factorization
        model = model.fit(self.rating_df)
        # get data for inferencing
        inferenceDF = self._create_inference_data(user_id, item_ids)
        # make inference
        return (
            model.transform(inferenceDF)
            .select(["item_id", "prediction"])
            .orderBy("prediction", ascending=False)
            .rdd.map(lambda r: (r[0], r[1]))
            .take(n_recommendations)
        )

    def _inference_user_id(self, model, user_id, item_ids, n_recommendations):
        """
        return top n movie recommendations based on user's input movie
        Parameters
        ----------
        model: spark ALS model
        iser_id: str, recommendtions for user
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar item_id recommendations
        """
        # get data for inferencing
        inferenceDF = self._create_inference_data(user_id, item_ids)
        # make inference
        return (
            model.transform(inferenceDF)
            .select(["item_id", "prediction"])
            .orderBy("prediction", ascending=False)
            .rdd.map(lambda r: (r[0], r[1]))
            .take(n_recommendations)
        )

    def make_recommendations_by_item_name(self, fav_item, n_recommendations):
        """
        make top n item recommendations
        Parameters
        ----------
        fav_item: str, name of user input item
        n_recommendations: int, top n recommendations
        """
        # make inference and get raw recommendations
        logger.info("Recommendation system start to make inference ...")
        t0 = time.time()
        raw_recommends = self._inference(
            self.model, fav_item, n_recommendations
        )
        item_ids = [r[0] for r in raw_recommends]
        scores = [r[1] for r in raw_recommends]
        logger.info(
            "It took my system {:.2f}s to make inference \n\
              ".format(
                time.time() - t0
            )
        )
        # get item names
        # item_names = self.item_df \
        #     .filter(col('item_id').isin(item_ids)) \
        #     .select('item_name') \
        #     .rdd.map(lambda r: r[0]) \
        #     .collect()
        # # print recommendations
        # logger.info('Recommendations for {}:'.format(fav_item))
        # for i in range(len(item_names)):
        #     logger.info(f'{i+1}: {item_names[i]}, with rating of {scores[i]}')

    def make_recommendations_by_user_id(self, user_id, item_ids, n_recommendations):
        """
        make top n item recommendations
        Parameters
        ----------
        fav_item: str, name of user input item
        n_recommendations: int, top n recommendations
        """
        # make inference and get raw recommendations
        logger.info("Recommendation system start to make inference ...")
        t0 = time.time()
        raw_recommends = self._inference_user_id(
            self.model, user_id, item_ids, n_recommendations
        )
        item_ids = [r[0] for r in raw_recommends]
        # scores = [r[1] for r in raw_recommends]
        logger.info(
            "It took my system {:.2f}s to make inference \n\
              ".format(
                time.time() - t0
            )
        )
        return item_ids
    

    def cosine_similarity(self, le, top_n, base_key, r_pipeline):
        logger.info(f"start process similarity by als with top_n: {top_n}, base_key: {base_key}")
        vecs = self.model.itemFactors.collect()
        item_feature_vecs = [e.features for e in vecs]
        item_id_index_list = [e.id for e in vecs]
        item_origin_id_list = le.inverse_transform(item_id_index_list)
        sim_matrix = pairwise.cosine_similarity(item_feature_vecs)
        sim_res = []
        logger.info(f"upload to redis")
        for i in range(len(item_origin_id_list)):
            main_item = item_origin_id_list[i]
            main_item_sim_list = []
            for j in range(len(item_origin_id_list)):
                if i == j:
                    continue
                sub_item = item_origin_id_list[j]
                main2sub_item_sim = sim_matrix[i][j]
                main_item_sim_list.append((sub_item, main2sub_item_sim))
            main_item_sim_list = sorted(main_item_sim_list, key=lambda x:x[1], reverse=True)[:top_n]
            upload_data = [f"{ssid}::{sim}" for ssid, sim in main_item_sim_list]
            r_pipeline.set(f"{base_key}_{main_item}", json.dumps(upload_data), ex=REDIS_EXPIRE_TIME)
            if i % 10000 == 0:
                r_pipeline.execute()
                logger.info(f"process als similarity {i/len(item_origin_id_list)*100}%, i: {i}")
        r_pipeline.execute()


def tune_ALS(model, train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    model: spark ML model, ALS
    train_data: spark DF with columns ['user_id', 'item_id', 'rating']
    validation_data: spark DF with columns ['user_id', 'item_id', 'rating']
    maxIter: int, max number of learning iterations
    regParams: list of float, one dimension of hyper-param tuning grid
    ranks: list of float, one dimension of hyper-param tuning grid
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float("inf")
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = model.setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(
                metricName="rmse",
                labelCol="rating",
                predictionCol="prediction",
            )
            rmse = evaluator.evaluate(predictions)
            logger.info(
                "{} latent factors and regularization = {}: "
                "validation RMSE is {}".format(rank, reg, rmse)
            )
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    logger.info(
        "\nThe best model has {} latent factors and "
        "regularization = {}".format(best_rank, best_regularization)
    )
    return best_model

@func_run_time_monitor
def _process(
    spark, site_id, item_type, act_types, mysql_conn
):
    """
    spark: spark instance
    site_id: str, for query data
    item_type: enumerate(item, product, coupon), for query condition
    act_types: List[str], for query condition
    """
    logger.info(f"start process {site_id}-{item_type} ALS")
    r_pipeline = redis_conn.conn.pipeline(transaction=False)
    if IS_DEV:
        if act_types and len(act_types) > 1:
            condition = f'and act_type in {tuple(act_types)}'
        elif act_types and len(act_types) == 1:
            condition = f'and act_type = "{act_types[0]}"'
        last_act_time = pd.read_sql(f'select max(act_time) from {schema}.rec_action where site_id = "{site_id}" and item_type = "{item_type}" {condition};', mysql_conn.conn)
        if not list(last_act_time['max(act_time)'])[0]:
            logger.info(f"{site_id}-{item_type} 无有效行为数据 ALS 模型无法训练")
            return
        alive_user_timestamp = list(last_act_time['max(act_time)'])[0] - RECALL_PARAMS["alive_time_delta"]
        act_time_bound = list(last_act_time['max(act_time)'])[0] - RECALL_PARAMS["read_data_time_delta"]
    else:
        alive_user_timestamp = time.time() - RECALL_PARAMS["alive_time_delta"]
        act_time_bound = time.time() - RECALL_PARAMS["read_data_time_delta"]
    act_df = get_act_data(
        mysql_conn.conn, site_id, item_type, act_types, act_time_bound
    )
    hot_recall(mysql_conn.conn, site_id, item_type, act_df)
    new_recall(mysql_conn.conn, site_id, item_type)
    logger.info(f"read act data complete total lens: {len(act_df)}")
    
    user_expose_items = get_expose_item_id(mysql_conn.conn, site_id, item_type)
    if item_type == 'coupon':
        type_rating_map = {
            'recive': 1,
            'certificate': 3,
        }
    else:
        type_rating_map = None
    rating_df = get_rating_by_df(act_df, type_rating_map)
    user_id_le = LabelEncoder()
    item_id_le = LabelEncoder()
    rating_df['user_id'] = user_id_le.fit_transform(rating_df['user_id'])
    rating_df['item_id'] = item_id_le.fit_transform(rating_df['item_id'])
    rating_df = spark.createDataFrame(rating_df)
    item_df = get_item_data(mysql_conn.conn, site_id, item_type)
    for k in item_df.keys():
        item_df[k] = item_df[k].astype(str)
    item_df = spark.createDataFrame(item_df)
    top_n = RECALL_PARAMS['base_recall'] + recall_n_calculate(act_df, RECALL_PARAMS["top_n_percent"])
    recommender = AlsRecommender(spark, item_df, rating_df)
    recommender.set_model_params(ALS_CONFIG['max_iter'], ALS_CONFIG['reg_param'], ALS_CONFIG['rank'])
    # sim_base_key = f"rec_sim4recall_{site_id}_{item_type}"
    sim_base_key = f"REC_SIM4RECALL_{site_id}_{item_type.upper()}"
    recommender.cosine_similarity(item_id_le, top_n, sim_base_key, r_pipeline)
    user_ids =  act_df[
        act_df["act_time"] > alive_user_timestamp
    ].user_id.unique()
    total_le_user_ids = [int(e) for e in list(user_id_le.transform(user_ids))]
    sub_user_df = rating_df[rating_df['user_id'].isin(total_le_user_ids)]
    user_recs = recommender.model.recommendForUserSubset(sub_user_df, top_n)
    candidates = user_recs.select('user_id', 'recommendations.item_id', 'recommendations.rating').collect()
    for i in range(len(candidates)):
        candidate = candidates[i]
        le_user_id = candidate[0]
        item_ids = candidate[1]
        item_ids = list(item_id_le.inverse_transform(item_ids))
        ori_user_id = user_id_le.inverse_transform([le_user_id])
        if not isinstance(ori_user_id, (int, str)):
            ori_user_id = ori_user_id[0]
        # user_expose_item_ids = list(user_expose_items[user_expose_items.user_id == ori_user_id].item_id)
        # for expose_item_id in user_expose_item_ids:
        #     item_ids.remove(expose_item_id)
        # key = f"rec_ALSrecall_{site_id}_{item_type}_{ori_user_id}"
        key = f"REC_ALSRECALL_{site_id}_{item_type.upper()}_{ori_user_id}"
        r_pipeline.set(key, json.dumps(item_ids), ex=REDIS_EXPIRE_TIME)
        if i % 10000 == 0:
            r_pipeline.execute()
            logger.info(f"{site_id}, {item_type}: process {i} of {len(total_le_user_ids)}: {i/len(total_le_user_ids)*100}%....")
    r_pipeline.execute()

@func_run_time_monitor
def process():
    conn = MysqlConn(**MysqlConfig)
    from offline.recall.utils import get_scene_info
    scene_info = get_scene_info(conn.conn)
    for site_id in scene_info:
        for item_type in scene_info[site_id]:
            _process(
                spark,
                site_id,
                item_type.lower(),
                ITEM_ACT_TYPES[item_type],
                conn
            )
