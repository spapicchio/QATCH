from unittest.mock import Mock, patch

import pandas as pd
import pytest

from qatch.database_reader import SingleDatabase
from qatch.sql_generator import GroupByGenerator


@pytest.fixture
def mock_database():
    mock_db = Mock(spec=SingleDatabase)
    mock_db.get_schema_given.return_value = Mock(name='schema', spec=pd.DataFrame)
    mock_db.get_table_given.return_value = Mock(name='table', spec=pd.DataFrame)
    return mock_db


@pytest.fixture
def group_by_generator(mock_database):
    generator = GroupByGenerator(mock_database)
    return generator


def test_simple_aggr_generator(group_by_generator):
    table_name = 'sample_table'

    # numerical and categorical attributes present
    with patch.object(group_by_generator, '_sample_cat_num_cols',
                      return_value=([], ['cat_col_1', 'cat_col_2'], ['num_col_1', 'num_col_2'])):
        generated_sql = group_by_generator.sql_generate(table_name)
        assert 'GROUPBY-NO-AGGR' in generated_sql['sql_tags']
        assert 'GROUPBY-COUNT' in generated_sql['sql_tags']
        assert 'GROUPBY-AGG-MAX' in generated_sql['sql_tags']
        # 2 for NO_AGGR, one for each possible combination of cat_col
        # 2 for GROUPBY COUNT, count len for each cat col
        # 16 tests for GROUPBY agg
        assert len(generated_sql['queries']) == 20
    # numerical attributes present
    with patch.object(group_by_generator, '_sample_cat_num_cols',
                      return_value=([], [], ['num_col_1', 'num_col_2'])):
        generated_sql = group_by_generator.sql_generate(table_name)
        # at least one column attribute is required (no groupby otherwise)
        assert len(generated_sql['sql_tags']) == 0

