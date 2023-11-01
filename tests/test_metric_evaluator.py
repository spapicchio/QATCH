import os
import shutil
import sqlite3

import pandas as pd
import pytest

from qatch import MetricEvaluator
from qatch.database_reader import MultipleDatabases, SingleDatabase

DB_PATH = 'test_db'
DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
TABLE_DICT = {'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}
TABLE_DATAFRAME = pd.DataFrame(TABLE_DICT)

TABLE_DICT_PRED = {'db_id': [f'{DB_NAME}_1', f'{DB_NAME}_2', f'{DB_NAME}_3'],
                   'query': [f'SELECT * FROM {TABLE_NAME}'] * 3,
                   'prediction': [f'SELECT * FROM {TABLE_NAME}'] * 3}

PREDICTION_DATAFRAME = pd.DataFrame(TABLE_DICT_PRED)


@pytest.fixture
def multiple_databases():
    os.makedirs(DB_PATH)
    # create 3 databases with the same table for the MultipleDatabase
    db_1 = SingleDatabase(db_path=DB_PATH, db_name=f'{DB_NAME}_1', tables={TABLE_NAME: TABLE_DATAFRAME})
    db_2 = SingleDatabase(db_path=DB_PATH, db_name=f'{DB_NAME}_2', tables={TABLE_NAME: TABLE_DATAFRAME})
    db_3 = SingleDatabase(db_path=DB_PATH, db_name=f'{DB_NAME}_3', tables={TABLE_NAME: TABLE_DATAFRAME})
    yield MultipleDatabases(DB_PATH)
    # Teardown: Close the databases and remove the temporary directory
    shutil.rmtree(DB_PATH)


@pytest.fixture
def metric_evaluator(multiple_databases):
    return MetricEvaluator(databases=multiple_databases)


def test_metric_evaluator_init(multiple_databases):
    evaluator = MetricEvaluator(databases=multiple_databases)
    assert evaluator.metrics == ['cell_precision', 'cell_recall',
                                 'tuple_cardinality', 'tuple_constraint',
                                 'tuple_order']

    evaluator = MetricEvaluator(databases=multiple_databases, metrics='tuple_order')
    assert evaluator.metrics == ['tuple_order']


def test_get_query_results_from_db(metric_evaluator):
    df_result, _ = metric_evaluator._get_query_results_from_db(PREDICTION_DATAFRAME)
    # assert that the prediction has been calculated
    assert 'query_result' in df_result.columns

    # assert that the prediction is correct
    query_result = [[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']]
    assert df_result['query_result'].tolist() == [query_result] * 3


def test_get_query_results_from_db_wrong_query(metric_evaluator):
    dict_pred = {'db_id': [f'{DB_NAME}_1', f'{DB_NAME}_2', f'{DB_NAME}_3'],
                 'query': [f'SELECT * FROM wrong_table_name'] * 3,
                 'prediction': [f'SELECT * FROM {TABLE_NAME}'] * 3}

    pred_df = pd.DataFrame(dict_pred)
    # check if raise an error
    with pytest.raises(sqlite3.OperationalError):
        metric_evaluator._get_query_results_from_db(pred_df)


def test_get_SP_query_results_from_db_equal(metric_evaluator):
    # the predictions are equal to the queries
    df, pred_col = metric_evaluator._get_SP_query_results_from_db(PREDICTION_DATAFRAME, 'prediction')
    assert df[pred_col].tolist() == ['EQUAL'] * len(PREDICTION_DATAFRAME)


def test_get_SP_query_results_from_db_different(metric_evaluator):
    """run the prediction query only if they are different to the queries"""
    pred_df = pd.DataFrame({'db_id': [f'{DB_NAME}_1'],
                            'query': [f'SELECT * FROM {TABLE_NAME}'],
                            'prediction': [f'SELECT "name" FROM {TABLE_NAME}']}
                           )
    # the predictions are different to the queries
    df, pred_col = metric_evaluator._get_SP_query_results_from_db(pred_df, 'prediction')
    target = [[x] for x in TABLE_DATAFRAME['name'].tolist()]
    assert df[pred_col][0] == target


def test_get_SP_query_results_from_db_error_query(metric_evaluator):
    """if the prediction query is wrong, return None"""
    pred_df = pd.DataFrame({'db_id': [f'{DB_NAME}_1'],
                            'query': [f'SELECT * FROM {TABLE_NAME}'],
                            'prediction': [f'SELECT "name" FROM error_table_name']}
                           )
    # the predictions are different to the queries
    df, pred_col = metric_evaluator._get_SP_query_results_from_db(pred_df, 'prediction')
    target = None
    assert df[pred_col][0] == target


@pytest.mark.parametrize("target, prediction, expected_result", [
    ("Hello, World!", "Hello, World!", True),  # Same strings, ignoring spaces
    ("Hello World", "HELLO WORLD", True),  # Case-insensitive match
    ("Hello, World", "Hello World", False),  # Different punctuation
    ("Hello World", "Hello, World!", False),  # Different punctuation
    ("(A, B)", " ( A , B)", True),  # Ignoring spaces around parentheses
    ("Hello", "Hello World", False),  # Different strings
    ("", "", True),  # Empty strings
])
def test_create_mask_target_equal_prediction(metric_evaluator, target, prediction, expected_result):
    output = metric_evaluator._create_mask_target_equal_prediction(target, prediction)
    assert output == expected_result


@pytest.mark.parametrize("predictions, expected_result", [
    ([f'SELECT * FROM {TABLE_NAME}'] * 3, []),
    ([f'SELECT "name" FROM {TABLE_NAME}'] * 3, ['cell_recall', 'tuple_constraint'])
])
def test_evaluate_with_df_no_order(metric_evaluator, predictions, expected_result):
    """expected result contains the metrics that are not 1.0 with the specified predictions"""
    # All equals without order by sql queries
    PREDICTION_DATAFRAME['prediction'] = predictions
    df = metric_evaluator.evaluate_with_df(PREDICTION_DATAFRAME, 'prediction', task='SP')
    metrics = metric_evaluator.metrics
    for metric in metrics:
        if metric == 'tuple_order':
            # always NONE in this case because no order
            assert df[metric].tolist() == [None] * len(PREDICTION_DATAFRAME)
            continue
        if metric not in expected_result:
            assert df[metric].tolist() == [1.0] * len(PREDICTION_DATAFRAME)
        else:
            assert df[metric].tolist() != [1.0] * len(PREDICTION_DATAFRAME)


@pytest.mark.parametrize("query, predictions, expected_result", [
    (
            [f'SELECT * FROM {TABLE_NAME} ORDER BY "name" ASC'] * 3,
            [f'SELECT * FROM {TABLE_NAME} ORDER BY "name" ASC'] * 3,  # predictions all equal
            []
    ),
    (
            [f'SELECT * FROM {TABLE_NAME} ORDER BY "name" DESC'] * 3,
            [f'SELECT * FROM {TABLE_NAME} ORDER BY "name" ASC'] * 3,  # predictions all different
            ['tuple_order']),
    (
            [f'SELECT * FROM {TABLE_NAME} ORDER BY "name" DESC'] * 3,
            [f'SELECT * FROM {TABLE_NAME} ORDER BY ASC'] * 3,  # wrong prediction
            # all the metrics are 0 because the prediction is None
            ['cell_precision', 'cell_recall', 'tuple_cardinality', 'tuple_constraint', 'tuple_order']
    )
])
def test_evaluate_with_df_with_order(metric_evaluator, query, predictions, expected_result):
    """expected result contains the metrics that are not 1.0 with the specified predictions"""
    # All equals without order by sql queries
    PREDICTION_DATAFRAME['prediction'] = predictions
    PREDICTION_DATAFRAME['query'] = query

    df = metric_evaluator.evaluate_with_df(PREDICTION_DATAFRAME, 'prediction', task='SP')
    metrics = metric_evaluator.metrics
    for metric in metrics:
        if metric not in expected_result:
            assert df[metric].tolist() == [1.0] * len(PREDICTION_DATAFRAME)
        else:
            assert df[metric].tolist() != [1.0] * len(PREDICTION_DATAFRAME)
