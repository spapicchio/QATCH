import os
import shutil

import pandas as pd
import pytest

from qatch.database_reader import MultipleDatabases, SingleDatabase
from qatch.test_generator import TestGenerator

DB_PATH = 'test_db'
DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
TABLE_DICT = {'id': [1, 2, 3], 'name': ['Alice', 'Bob', None]}
TABLE_DATAFRAME = pd.DataFrame(TABLE_DICT)

TABLE_DICT_PRED = {'db_id': [f'{DB_NAME}_1', f'{DB_NAME}_2', f'{DB_NAME}_3'],
                   'query': [f'SELECT * FROM {TABLE_NAME}'] * 3,
                   'prediction': [f'SELECT * FROM {TABLE_NAME}'] * 3}

PREDICTION_DATAFRAME = pd.DataFrame(TABLE_DICT_PRED)


@pytest.fixture
def multiple_databases():
    os.makedirs(DB_PATH)
    # create 3 databases with the same table for the MultipleDatabas
    db_1 = SingleDatabase(db_path=DB_PATH, db_name=f'{DB_NAME}_1', tables={TABLE_NAME: TABLE_DATAFRAME})
    db_2 = SingleDatabase(db_path=DB_PATH, db_name=f'{DB_NAME}_2', tables={TABLE_NAME: TABLE_DATAFRAME})
    db_3 = SingleDatabase(db_path=DB_PATH, db_name=f'{DB_NAME}_3', tables={TABLE_NAME: TABLE_DATAFRAME})
    yield MultipleDatabases(DB_PATH)
    # Teardown: Close the databases and remove the temporary directory
    shutil.rmtree(DB_PATH)


@pytest.fixture
def init_test_generator(multiple_databases) -> TestGenerator:
    return TestGenerator(databases=multiple_databases)


@pytest.mark.parametrize("generators, db_names,  expected_result", [
    (None, None, (['select', 'orderby', 'distinct', 'where', 'groupby',
                   'having', 'simpleAgg', 'nullCount'],
                  [f'{DB_NAME}_1', f'{DB_NAME}_2', f'{DB_NAME}_3'])
     ),
    ('select', f'{DB_NAME}_1', (['select'],
                                [f'{DB_NAME}_1'])
     ),
    (['select', 'orderby'], [f'{DB_NAME}_1', f'{DB_NAME}_2'], (['select', 'orderby'],
                                                               [f'{DB_NAME}_1', f'{DB_NAME}_2'])
     ),
])
def test_init_parameters(init_test_generator, generators, db_names, expected_result):
    generators, db_names = init_test_generator._init_params(generators=generators,
                                                            db_names=db_names)
    # Check if the generators and db_names are as expected
    assert set(generators) == set(expected_result[0])
    assert set(db_names) == set(expected_result[1])


@pytest.mark.parametrize("generators, db_names", [
    ('wrong', f'{DB_NAME}_1'),
    ('select', 'wrong'),
])
def test_init_parameters_error(init_test_generator, generators, db_names):
    with pytest.raises(KeyError):
        generators, db_names = init_test_generator._init_params(generators=generators,
                                                                db_names=db_names)


@pytest.mark.parametrize("generators, db_names", [
    (['nullCount'], [f'{DB_NAME}_1']),
    (None, None),
    (['select', 'orderby'], [f'{DB_NAME}_1', f'{DB_NAME}_2']),
    (None, [f'{DB_NAME}_1']),
    (['select'], None)
])
def test_generate(init_test_generator, generators, db_names):
    tests_df = init_test_generator.generate(generators=generators, db_names=db_names)
    if db_names is None:
        db_names = [f'{DB_NAME}_1', f'{DB_NAME}_2', f'{DB_NAME}_3']
    if generators is None:
        generators = ['select', 'orderby', 'distinct', 'where', 'groupby',
                      'having', 'simpleAgg', 'nullCount']
    for db_name in db_names:
        assert db_name in tests_df.db_id.unique().tolist()
    for generator in generators:
        if generator == 'simpleAgg':
            generator = "simple-agg"
        if generator == 'nullCount':
            generator = "null"
        assert tests_df.sql_tags.str.contains(generator.upper()).any()
