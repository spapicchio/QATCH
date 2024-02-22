import pandas as pd
import pytest

from qatch import MetricEvaluator
from qatch.database_reader import MultipleDatabases, SingleDatabase

DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
TABLE_DICT = {'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}
TABLE_DATAFRAME = pd.DataFrame(TABLE_DICT)

TABLE_DICT_PRED = {'db_id': [f'{DB_NAME}_1', f'{DB_NAME}_2', f'{DB_NAME}_3'],
                   'query': [f'SELECT * FROM {TABLE_NAME}'] * 3,
                   'prediction': [f'SELECT * FROM {TABLE_NAME}'] * 3}

PREDICTION_DATAFRAME = pd.DataFrame(TABLE_DICT_PRED)


@pytest.fixture
def multiple_databases(tmp_path):
    # create 3 databases with the same table for the MultipleDatabase
    db_1 = SingleDatabase(db_path=tmp_path, db_name=f'{DB_NAME}_1', tables={TABLE_NAME: TABLE_DATAFRAME})
    db_2 = SingleDatabase(db_path=tmp_path, db_name=f'{DB_NAME}_2', tables={TABLE_NAME: TABLE_DATAFRAME})
    db_3 = SingleDatabase(db_path=tmp_path, db_name=f'{DB_NAME}_3', tables={TABLE_NAME: TABLE_DATAFRAME})
    yield MultipleDatabases(tmp_path)


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


def test_get_SP_query_results_from_db_equal(metric_evaluator):
    # the predictions are equal to the queries
    dict_pred = {'db_id': f'{DB_NAME}_1',
                 'query': f'SELECT * FROM  table ORDER BY "name" ASC',
                 'prediction': f'SELECT * FROM  table ORDER BY "name" ASC'}
    output = metric_evaluator.evaluate_single_test_SP(dict_pred, 'prediction', 'query')
    assert all([val == 1 for x, val in output.items()])


def test_get_SP_query_results_from_db_error_query(metric_evaluator):
    """if the prediction query is wrong, return 0"""
    dict_pred = {'db_id': f'{DB_NAME}_1',
                 'query': f'SELECT * FROM  table ORDER BY "name" ASC',
                 'prediction': f'SELECT * FROM  wrong_table ORDER BY "name" ASC'}
    output = metric_evaluator.evaluate_single_test_SP(dict_pred, 'prediction', 'query')
    assert all([val == 0 for x, val in output.items()])


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
    output = metric_evaluator.are_cleaned_sql_identical(target, prediction)
    assert output == expected_result


@pytest.mark.parametrize("predictions, expected_result", [
    ([f'SELECT * FROM {TABLE_NAME}'] * 3, []),
    ([f'SELECT "name" FROM {TABLE_NAME}'] * 3, ['cell_recall_prediction', 'tuple_constraint_prediction'])
])
def test_evaluate_with_df_no_order(metric_evaluator, predictions, expected_result):
    """expected result contains the metrics that are not 1.0 with the specified predictions"""
    # All equals without order by sql queries
    PREDICTION_DATAFRAME['prediction'] = predictions
    df = metric_evaluator.evaluate_with_df(PREDICTION_DATAFRAME, 'prediction', task='SP')
    metrics = metric_evaluator.metrics
    for metric in metrics:
        metric = f'{metric}_prediction'
        if metric == 'tuple_order_prediction':
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
            ['tuple_order_prediction']),
    (
            [f'SELECT * FROM {TABLE_NAME} ORDER BY "name" DESC'] * 3,
            [f'SELECT * FROM {TABLE_NAME} ORDER BY ASC'] * 3,  # wrong prediction
            # all the metrics are 0 because the prediction is None
            ['cell_precision_prediction', 'cell_recall_prediction',
             'tuple_cardinality_prediction', 'tuple_constraint_prediction',
             'tuple_order_prediction']
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
        metric = f'{metric}_prediction'
        if metric not in expected_result:
            assert df[metric].tolist() == [1.0] * len(PREDICTION_DATAFRAME)
        else:
            assert df[metric].tolist() != [1.0] * len(PREDICTION_DATAFRAME)
