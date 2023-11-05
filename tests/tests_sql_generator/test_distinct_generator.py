from unittest.mock import patch, Mock

import pandas as pd
import pytest

from qatch.database_reader import SingleDatabase
from qatch.sql_generator import DistinctGenerator


@pytest.fixture
def mock_database():
    mock_db = Mock(spec=SingleDatabase)
    mock_db.get_schema_given.return_value = Mock(name='schema', spec=pd.DataFrame)
    mock_db.get_table_given.return_value = Mock(name='table', spec=pd.DataFrame)
    return mock_db


def test_distinct_generator(mock_database):
    generator = DistinctGenerator(mock_database)
    table_name = 'example_table'

    with patch.object(generator, '_sample_cat_num_cols', return_value=([], ['col1', 'col2'], [])):
        generated_sql = generator.sql_generate(table_name)

    assert 'DISTINCT-SINGLE' in generated_sql['sql_tags']
    assert 'DISTINCT-MULT' in generated_sql['sql_tags']

    assert len(generated_sql['queries']) == 4  # two for single column and two for multiple columns
    assert len(generated_sql['questions']) == 4

    single_col_query = f'SELECT DISTINCT "col1" FROM "{table_name}"'
    mult_col_query = f'SELECT DISTINCT "col2", "col1" FROM "{table_name}"'

    assert single_col_query in generated_sql['queries']
    assert mult_col_query in generated_sql['queries']

    single_col_question = f'Show the different "col1" in the table {table_name}'
    mult_col_question = f'Show the different "col2", "col1" in the table "{table_name}"'

    assert single_col_question in generated_sql['questions']
    assert mult_col_question in generated_sql['questions']
