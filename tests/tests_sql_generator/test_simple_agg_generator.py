from unittest.mock import Mock, patch

import pandas as pd
import pytest

from qatch.database_reader import SingleDatabase
from qatch.sql_generator import SimpleAggGenerator


@pytest.fixture
def mock_database():
    mock_db = Mock(spec=SingleDatabase)
    mock_db.get_schema_given.return_value = Mock(name='schema', spec=pd.DataFrame)
    mock_db.get_table_given.return_value = Mock(name='table', spec=pd.DataFrame)
    return mock_db


@pytest.fixture
def simple_aggr_generator(mock_database):
    generator = SimpleAggGenerator(mock_database)
    return generator


def test_simple_aggr_generator(simple_aggr_generator):
    table_name = "sample_table"
    # numerical and categorical attributes present
    with patch.object(simple_aggr_generator, '_sample_cat_num_cols',
                      return_value=([], ['cat_col_1', 'cat_col_2'], ['num_col_1', 'num_col_2'])):
        generated_sql = simple_aggr_generator.sql_generate(table_name)

        assert 'SIMPLE-AGG-COUNT' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-COUNT-DISTINCT' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-MAX' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-MIN' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-AVG' in generated_sql['sql_tags']
        # 1 count general
        # 1 count distinct for each categorical
        # 3 [max, min, avg] for each numerical
        assert len(generated_sql['queries']) == 9

    # only numerical
    with patch.object(simple_aggr_generator, '_sample_cat_num_cols',
                      return_value=([], [], ['num_col_1', 'num_col_2'])):
        generated_sql = simple_aggr_generator.sql_generate(table_name)

        assert 'SIMPLE-AGG-COUNT' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-COUNT-DISTINCT' not in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-MAX' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-MIN' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-AVG' in generated_sql['sql_tags']
        assert len(generated_sql['queries']) == 7
    # only categorical
    with patch.object(simple_aggr_generator, '_sample_cat_num_cols',
                      return_value=([], ['cat_col_1', 'cat_col_2'], [])):
        generated_sql = simple_aggr_generator.sql_generate(table_name)

        assert 'SIMPLE-AGG-COUNT' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-COUNT-DISTINCT' in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-MAX' not in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-MIN' not in generated_sql['sql_tags']
        assert 'SIMPLE-AGG-AVG' not in generated_sql['sql_tags']
        assert len(generated_sql['queries']) == 3
