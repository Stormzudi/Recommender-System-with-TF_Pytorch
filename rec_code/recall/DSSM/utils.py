# coding: utf-8
import uuid
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import json

from common.log_utils import Loggers
from setting.default import DSSM_CONFIG, MysqlConfig

schema = MysqlConfig['db']
logger = Loggers("DSSM utils").logger


def get_model_feature_names(item_type):

    user_sparse_features = DSSM_CONFIG[item_type]["features"][
        "user_sparse_features"
    ]
    user_dense_features = DSSM_CONFIG[item_type]["features"][
        "user_dense_features"
    ]
    item_sparse_features = DSSM_CONFIG[item_type]["features"][
        "item_sparse_features"
    ]
    item_dense_features = DSSM_CONFIG[item_type]["features"][
        "item_dense_features"
    ]

    return (
        user_sparse_features,
        user_dense_features,
        item_sparse_features,
        item_dense_features,
    )


def get_user_data(site_id, conn):
    """
    Params:
        site_id: 此次处理的site标识
        conn: 数据源连接
    Return:
        DataFrame

    暂用pandasDataFrame，后续版本更新至sparkDataFrame
    """
    logger.debug(f"start get {site_id} user data")
    sql = f"""
    SELECT user_id, age, level_id, gender
    FROM {schema}.rec_user
    WHERE 
        site_id = '{site_id}'
        and
        is_deleted = 0
    ;
    """
    user_df = pd.read_sql(sql, conn)
    return user_df


def get_action_data(site_id, item_type, conn):
    """
    Params:
        site_id: 此次处理的site标识
        conn: 数据源连接
    Return:
        DataFrame

    暂用pandasDataFrame，后续版本更新至sparkDataFrame
    """
    logger.debug(f"start get {site_id} action data")
    if item_type == 'coupon':
        act_type_condition = "act_type in ('recive', 'certificate')"
    else:
        act_type_condition = "act_type = 'buy'"
    sql = f"""
    SELECT DISTINCT user_id, item_id, act_time
    from 
        {schema}.rec_action
    WHERE
        site_id = '{site_id}'
        and
        {act_type_condition}
        and 
        item_type = '{item_type}'
        and
        is_deleted = 0
    ;
    """
    action_df = pd.read_sql(sql, conn)
    return action_df


def get_product_data(site_id, conn):
    """
    Params:
        site_id: 此次处理的site标识
        conn: 数据源连接
    Return:
        DataFrame

    暂用pandasDataFrame，后续版本更新至sparkDataFrame
    """
    logger.debug(f"start get {site_id} user data")
    sql = f"""
    SELECT DISTINCT item_id, store_id, category1_id
    FROM {schema}.rec_product
    WHERE 
        site_id = '{site_id}'
        and
        is_deleted = 0
    ;
    """
    product_df = pd.read_sql(sql, conn)
    return product_df


def get_store_data(site_id, conn):
    """
    Params:
        site_id: 此次处理的site标识
        conn: 数据源连接
    Return:
        DataFrame

    v暂用pandasDataFrame，后续版本更新至sparkDataFrame
    """
    logger.debug(f"start get {site_id} store data")
    sql = f"""
    select
        DISTINCT item_id, category1_id
    from
        {schema}.rec_store
    where
        site_id = '{site_id}'
        and
        is_deleted = 0
    ;
    """

    store_df = pd.read_sql(sql, conn)
    return store_df


def get_coupon_data(site_id, conn):
    sql = f"""
    select 
        item_id, item_value, get_value,
        use_value, item_type,
        category1_id
    from
        {schema}.rec_coupon
    where
        site_id = "{site_id}"
        and is_deleted = 0
    ;
    """
    return pd.read_sql(sql, conn)


def clear_exists_embs(conn, site_id, table, version, ids, id_type):
    logger.info(f"clear {site_id}, {table}, {len(ids)} {datetime.now().strftime('%Y%m%d')}, embs")
    condition = ''
    if len(ids) == 1:
        condition = f"{id_type} = {ids[0]}"
    else:
        condition = f"{id_type} in {tuple(ids)}"
    try:
        with conn.cursor() as cursor:
            sql = f"""
            delete from {table}
            where 
                {condition}
                and version = '{version}'
                and dt = "{datetime.now().strftime('%Y%m%d')}"
                and site_id = "{site_id}"
            """
            res = cursor.execute(sql)
            logger.info(f"{site_id}, {table}, {datetime.now().strftime('%Y%m%d')} clear {res} rows")
    except Exception as exc:
        logger.error(f"table: {table} clear_exists_embs error!exc_info=", exc_info=exc)
    finally:
        conn.commit()


def insert_embedding(conn, site_id, table, version, embs, index_to_id_dict, id_to_index_dict, id_type, step=5000):
    try:
        with conn.cursor() as cursor:
            sql = f"""
            insert into {table} (id, {id_type}, {id_type}_index, vec, dt, site_id, version)
            values (%s, %s, %s, %s, %s, %s, %s)
            """
            insert_data_list = []
            for i, emb in enumerate(embs):
                insert_data_list.append(
                    (
                    str(uuid.uuid1()), 
                    index_to_id_dict[i], 
                    id_to_index_dict[index_to_id_dict[i]],
                    json.dumps(emb.tolist()),
                    datetime.now().strftime('%Y%m%d'),
                    site_id,
                    version,
                    )
                )
                if len(insert_data_list) >= step:
                    cursor.executemany(sql, insert_data_list)
                    insert_data_list = []
            if insert_data_list:
                cursor.executemany(sql, insert_data_list)
            conn.commit()
    except Exception as exc:
        logger.error(f"table: {table} insert_embedding error!exc_info=", exc_info=exc)
    finally:
        conn.commit()


def get_raw_data(site_id, item_type, mysql_con):
    """预处理数据集"""
    user = get_user_data(site_id, mysql_con)
    user_sparse_features, user_dense_features, item_sparse_features, item_dense_features = get_model_feature_names(item_type)
    sparse_features = user_sparse_features + item_sparse_features
    dense_features = user_dense_features + item_dense_features
    user_features = user_sparse_features + user_dense_features
    item_features = item_sparse_features + item_dense_features
    if item_type == 'product':
        item = get_product_data(site_id, mysql_con)
    elif item_type == 'store':
        item = get_store_data(site_id, mysql_con)
    elif item_type == 'coupon':
        item = get_coupon_data(site_id, mysql_con)
    event = get_action_data(site_id, item_type, mysql_con)

    # 数据预处理
    logger.debug(f"start process {site_id} user feature ")
    user = user.fillna(-1)
    event['datetime'] = event['act_time'].apply(lambda x: datetime.fromtimestamp(x))
    event_data = event.sort_values(by='datetime', ascending=True).reset_index(drop=True)
    data = pd.merge(event_data, user, how='inner', on='user_id')
    data = pd.merge(data, item, how='inner', on='item_id')
    sparse_features_original = list(map(lambda x: x.replace('_index', ''), sparse_features))
    # 离散特征字符串化, 再转化为index
    data[sparse_features_original] = data[sparse_features_original].astype(str)
    feature_dict = {}
    for feat in sparse_features + ['user_id_index', 'item_id_index']:
        lbe = LabelEncoder()
        feat_original = feat.replace('_index', '')
        data[feat] = lbe.fit_transform(data[feat_original]) + 1
        index = list(range(1, len(lbe.classes_) + 1))
        # original value : index
        feature_dict[feat_original] = dict(zip(lbe.classes_, index))

    for feat in sparse_features + ['user_id_index', 'item_id_index']:
        if feat in user_features + ['user_id_index']:
            feat_original = feat.replace('_index', '')
            user[feat] = user[feat_original].apply(lambda x: feature_dict[feat_original].get(str(x), 0))
        elif feat in item_features + ['item_id_index']:
            feat_original = feat.replace('_index', '')
            item[feat] = item[feat_original].apply(lambda x: feature_dict[feat_original].get(str(x), 0))
    data = data.sort_values(by='datetime', ascending=True).reset_index(drop=True)
    # import pdb;pdb.set_trace()
    return user, item, data, feature_dict
