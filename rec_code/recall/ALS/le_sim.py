# coding: utf-8
import json
from common.log_utils import Loggers
import os

import pandas as pd
from sklearn.metrics import pairwise

from common.database import mysql_conn, redis_conn
from setting.default import (
    BASE_LE_PKL_SAVE_PATH,
    IS_DEV,
    PROCESS_SITE_IDS,
    REDIS_STORE_FEATURES,
    REDIS_USER_FEATURES,
    REDIS_EXPIRE_TIME,
)

from offline.recall.utils import get_item_feature

logger = Loggers('recall_processer_v1').logger


def _process_item_sim(site_id, item_type):
    """
    2022-01-26 基于label_encoder
    """
    item_vec_data = get_item_feature(mysql_conn.conn, site_id, item_type)
    index_to_item_id = {}
    vec_list = []
    sim_res = []
    for i, rows in item_vec_data.iterrows():
        vec_list.append(json.loads(rows.vec))
        index_to_item_id[i] = rows.item_id
    sim_matrix = pairwise.cosine_similarity(vec_list)
    logger.info(f"site_id: {site_id},total {item_type}: {len(vec_list)}")

    logger.info(f"site_id: {site_id} format sim_relation")
    for i in range(len(vec_list)):
        main_store = index_to_item_id[i]
        main_item_sim_list = []
        for j in range(len(vec_list)):
            if i == j:
                continue
            sub_store = index_to_item_id[j]
            main2sub_item_sim = sim_matrix[i][j]
            main_item_sim_list.append((sub_store, main2sub_item_sim))
        main_item_sim_list = sorted(
            main_item_sim_list, key=lambda x: x[1], reverse=True
        )
        sim_res.append((main_store, main_item_sim_list))

    logger.info(f"site_id: {site_id} {item_type} insert into redis")
    for item_id, sims in sim_res:
        sim_list = [f"{ssid}::{sim}" for ssid, sim in sims]
        key = f"rec_sim4recall_{site_id}_{item_type}_{item_id}"
        redis_conn.conn.set(key, json.dumps(sim_list), ex=REDIS_EXPIRE_TIME)


def process():
    for site_id, item_types in SITE_ITEM_TYPES.items():
        for item_type in item_types:
            _process_item_sim(
                site_id,
                item_type
            ) 
