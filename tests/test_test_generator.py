import pandas as pd
import pytest

from qatch.database_reader import MultipleDatabases, SingleDatabase
from qatch.test_generator import TestGenerator

DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
TABLE_DICT = {
    'column1': [1, 2, 3, 4],
    'column2': ['A', 'B', 'A', 'C'],
    'column3': [10.5, 20.3, 15.2, 18.7],
    'column4': ['A', 'B', 'A', 'C'],
    'column5': ['A', 'B', 'A', None],
    'column6': [10.5, 20.3, 15.2, 18.7],
    'column7': [10.5, 20.3, 15.2, 18.7],
    'column8': ['A', 'B', 'A', 'C'],
    'col_id': [1111, 2222, 3333, 4444],
}
TABLE_DATAFRAME = pd.DataFrame(TABLE_DICT)

TABLE_DICT_PRED = {'db_id': [f'{DB_NAME}_1', f'{DB_NAME}_2', f'{DB_NAME}_3'],
                   'query': [f'SELECT * FROM {TABLE_NAME}'] * 3,
                   'prediction': [f'SELECT * FROM {TABLE_NAME}'] * 3}

PREDICTION_DATAFRAME = pd.DataFrame(TABLE_DICT_PRED)


@pytest.fixture
def multiple_databases(tmp_path):
    tmp_path = str(tmp_path)
    # create 3 databases with the same table for the MultipleDatabas
    _ = SingleDatabase(db_path=tmp_path, db_name=f'{DB_NAME}_1', tables={TABLE_NAME: TABLE_DATAFRAME})
    _ = SingleDatabase(db_path=tmp_path, db_name=f'{DB_NAME}_2', tables={TABLE_NAME: TABLE_DATAFRAME})
    _ = SingleDatabase(db_path=tmp_path, db_name=f'{DB_NAME}_3', tables={TABLE_NAME: TABLE_DATAFRAME})
    _ = SingleDatabase(db_path=tmp_path,
                       db_name=f'{DB_NAME}_4',
                       tables={TABLE_NAME: TABLE_DATAFRAME,
                               'test_table_2': pd.DataFrame({'col_id': [2222], 'col2': ['A']}
                                                            )
                               }
                       )
    return MultipleDatabases(tmp_path)


@pytest.fixture
def test_generator(multiple_databases) -> TestGenerator:
    return TestGenerator(databases=multiple_databases)


@pytest.mark.parametrize("generators, db_names,  expected_result", [
    (None, None, (['select', 'orderby', 'distinct', 'where', 'groupby',
                   'having', 'simpleAgg', 'nullCount', 'join'],
                  [f'{DB_NAME}_1', f'{DB_NAME}_2', f'{DB_NAME}_3', f'{DB_NAME}_4'])
     ),
    ('select', f'{DB_NAME}_1', (['select'],
                                [f'{DB_NAME}_1'])
     ),
    (['select', 'orderby'], [f'{DB_NAME}_1', f'{DB_NAME}_2'], (['select', 'orderby'],
                                                               [f'{DB_NAME}_1', f'{DB_NAME}_2'])
     ),
])
def test_init_parameters(test_generator, generators, db_names, expected_result):
    generators, db_names = test_generator._init_params(generators=generators,
                                                       db_names=db_names)
    # Check if the generators and db_names are as expected
    assert set(generators) == set(expected_result[0])
    assert set(db_names) == set(expected_result[1])


@pytest.mark.parametrize("generators, db_names", [
    ('wrong', f'{DB_NAME}_1'),
    ('select', 'wrong'),
])
def test_init_parameters_error(test_generator, generators, db_names):
    with pytest.raises(KeyError):
        generators, db_names = test_generator._init_params(generators=generators,
                                                           db_names=db_names)


@pytest.mark.parametrize("generators, db_names", [
    (['nullCount'], [f'{DB_NAME}_1']),
    (None, None),
    (['select', 'orderby'], [f'{DB_NAME}_1', f'{DB_NAME}_2']),
    (None, [f'{DB_NAME}_1']),
    (['select'], None),
    (['join'], [f'{DB_NAME}_4']),

])
def test_generate(test_generator, generators, db_names):
    tests_df = test_generator.generate(generators=generators, db_names=db_names)
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


@pytest.mark.parametrize("seed", [2023, 43, 1, 4, 293])
def test_generate_equal_seed(test_generator, seed):
    tests_df_1 = test_generator.generate(generators=None, db_names=None, seed=seed)
    tests_df_2 = test_generator.generate(generators=None, db_names=None, seed=seed)
    compare_df = tests_df_1 == tests_df_2
    assert all(compare_df.all().tolist())


@pytest.mark.parametrize("seed", [2023, 43, 1, 4, 293])
def test_generate_different_seed(test_generator, seed):
    tests_df_1 = test_generator.generate(generators=None, db_names=None, seed=seed)
    tests_df_2 = test_generator.generate(generators=None, db_names=None, seed=None)
    compare_df = tests_df_1 == tests_df_2
    assert not all(compare_df.all().tolist())
