import json
import sqlite3

import pandas as pd

from src import Tapas, Tapex, ChatGPT, TestGenerator, MetricEvaluator, TestGeneratorSpider
from utils_preprocess_data import read_data


def init_model(model_name):
    name = _get_name_from_model_path(model_name)
    if name == 'tapas':
        model = Tapas(model_name)
    elif name == 'tapex':
        model = Tapex(model_name)
    elif name == 'chatgpt':
        # open credentials json file
        with open('credentials.json', 'r') as f:
            credentials = json.load(f)
        model = ChatGPT(api_key=credentials.api_key_chatgpt,
                        api_org=credentials.api_org_chatgpt)
    return model


def init_test_generator(db_category, model_name, db_base_path, table_base_path):
    name = _get_name_from_model_path(model_name)
    db_tables = read_data(name=db_category, model_name=name, base_path=table_base_path)

    test_generator = TestGenerator(
        db_save_path=f'{db_base_path}/{db_category}',
        db_name=db_category,
        db_tables=db_tables
    )
    return test_generator


def init_test_generator_spider(spider_base_path):
    return TestGeneratorSpider(spider_base_path=spider_base_path)


def init_metric_evaluator(metrics):
    return MetricEvaluator(metrics=metrics)


def get_summary(df, metrics):
    if metrics is None:
        metrics = ['cell_precision', 'cell_recall', 'tuple_cardinality',
                   'tuple_constraint', 'tuple_order']
    out = {metric: round(df[metric].mean(), 2) for metric in metrics}
    return pd.DataFrame.from_dict(out, orient='index', columns=['value'])


def _get_name_from_model_path(model_name):
    if 'tapas' in model_name:
        name = 'tapas'
    elif 'tapex' in model_name:
        name = 'tapex'
    elif 'chatgpt' in model_name:
        name = 'chatgpt'
    elif 'sp' in model_name:
        name = 'sp'
    else:
        raise ValueError(f'Model {model_name} not found. '
                         f'Available models are [tapas, tapex, chatgpt, sp]')
    return name


def get_predictions_results_from_dbs(base_path_db: str, df: pd.DataFrame, predictions: str):
    def get_results_from_db(db_id, queries):
        path = f'{base_path_db}/{db_id}/{db_id}/{db_id}.sqlite'
        # sqlite3 connection
        conn = sqlite3.connect(path)
        # get the results
        return [conn.execute(query).fetchall() for query in queries]

    # 1. group the df by db_id
    grouped_df = df.groupby('db_id').agg(list)
    grouped_df['query_result_predictions'] = grouped_df.apply(
        lambda row: get_results_from_db(row.name, row[predictions]),
        axis=1
    )
    return grouped_df.explode(['tbl_name', 'sql_tags', 'query', 'question',
                               'query_result', 'query_result_predictions'])
