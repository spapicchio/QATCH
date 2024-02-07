from unittest.mock import Mock

import pandas as pd
import pytest

from qatch.sql_generator import JoinGenerator


def side_effect_cat_cols(table_name, value=None):
    if value:
        return (None, ['col2'], None) if table_name == 'tbl_1' else (None, ['col4'], None)

    if table_name == 'tbl_1':
        return None, ['col1', 'col2', 'col_ID1', 'col_ID2'], None
    if table_name == 'tbl_2':
        return None, ['col3', 'col4', 'col_ID1', 'col_ID2'], None


def side_effect_get_schema(table_name):
    if table_name == 'tbl_1':
        return pd.DataFrame({'name': ['col1', 'col2', 'col_ID1', 'col_ID2']})
    if table_name == 'tbl_2':
        return pd.DataFrame({'name': ['col3', 'col4', 'col_ID1', 'col_ID2']})


@pytest.fixture
def mock_single_database():
    single_database = Mock()
    single_database.table_names = ['tbl_1', 'tbl_2']
    single_database.get_schema_given.side_effect = side_effect_get_schema
    return single_database


@pytest.fixture
def join_generator(mock_single_database):
    generator = JoinGenerator(mock_single_database)
    generator._sample_cat_num_cols = Mock().side_effect = side_effect_cat_cols
    return generator


@pytest.mark.parametrize("t1_cols, t2_cols, result", [
    (['col1', 'col2', 'col_ID'], ['col1', 'col2', 'col_ID'], ['col_ID']),
    (['col1', 'col2', 'col_ID'], ['col1', 'col2'], []),
    (['col1', 'col2'], ['col1', 'col2', 'col_ID'], []),
    (['col1', 'col2', 'col_ID1', 'col_ID2'], ['col1', 'col2', 'col_ID1'], ['col_ID1']),
    (['col1', 'col2', 'col_ID1', 'col_ID2'], ['col1', 'col2', 'col_ID1', 'col_ID2'], ['col_ID1', 'col_ID2']),
])
def test_get_columns_to_join(t1_cols, t2_cols, result):
    assert result == JoinGenerator._get_columns_to_join(t1_cols, t2_cols)


def test_get_table_name_to_join(join_generator):
    result = join_generator._get_table_name_to_join('tbl_1')
    assert result
    assert 'tbl_2' in result.keys()
    assert 'tbl_1' not in result.keys()
    assert set(result['tbl_2']) == {'col_ID1', 'col_ID2'}


def test_generate_join_project_all(join_generator):
    table_to_join2cols = join_generator._get_table_name_to_join('tbl_1')
    join_generator._generate_join_project_all('tbl_1', table_to_join2cols)
    result = join_generator.sql_generated
    assert len(result['queries']) == len(result['questions']) == len(result['sql_tags']) == 2
    assert set(result['sql_tags']) == {'JOIN-PROJECT-ALL'}


def test_generate_join_cat_columns(join_generator):
    table_to_join2cols = join_generator._get_table_name_to_join('tbl_1')
    join_generator._generate_join_cat_columns('tbl_1', table_to_join2cols)
    result = join_generator.sql_generated
    # select one random categorical column for each possible join
    assert len(result['queries']) == len(result['questions']) == len(result['sql_tags']) == 2
    assert set(result['sql_tags']) == {'JOIN-PROJECT-CAT'}