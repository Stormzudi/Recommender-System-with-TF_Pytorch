# coding: utf-8
import pandas as pd
from common.log_utils import Loggers
from setting.default import MysqlConfig

logger = Loggers('recall_processer_v2').logger
schema = MysqlConfig['db']

def get_expose_item_id(conn, site_id, item_type):
    '''
    '''
    sql = f"""
    select 
        user_id, item_id
    from
        {schema}.rec_history
    where
        scene_id = "{site_id}"
        and item_type = "{item_type}"
    ;
    """
    return pd.read_sql(sql, conn)


def get_act_2_rating_bysql(conn, site_id, item_type):
    sql = f"""
    select 
        user_id, item_id, count(1) as rating
    from
        {schema}.rec_action
    where
        site_id = "{site_id}"
        and
        act_type = 'buy'
        and
        item_type = "{item_type}"
    group by
        user_id, item_id
    ;
    """
    df = pd.read_sql(sql)
    return df


def get_rating_by_df(df, type_rating_map=None):
    '''
    df:
        user_id, item_id, act_time, act_type
        1,a,127911313,buy
        2,a,127911313,buy
        1,b,127911313,view
        3,c,127911313,buy
        ...
    type_rating_map:
        {
            act_type: rating_weight
        }
    return:
        user_id, item_id, rating
        1,a,3
        1,b,1
        2,a,3
        3,c,3
    '''
    if not type_rating_map:
        type_rating_map = {'buy': 1}
    df['counts'] = 1
    df = df[df.act_type.isin(type_rating_map.keys())].groupby(['user_id', 'item_id', 'act_type']).agg('sum').reset_index()
    df['rating'] = df.apply(lambda x:x['counts'] * type_rating_map[x['act_type']], axis=1)
    df = df[['user_id', 'item_id', 'rating']]
    return df
