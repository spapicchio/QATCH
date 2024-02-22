from unittest.mock import patch

import pandas as pd
import pytest

from qatch.database_reader import SingleDatabase
from qatch.sql_generator import NullGenerator

# Define test data
DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
TABLE_DICT = {'id': [1, None, 3], 'name': ['Alice', 'Bob', None]}
TABLE_DATAFRAME = pd.DataFrame(TABLE_DICT)


# Fixture to create a temporary SingleDatabase object for testing
@pytest.fixture
def single_database(tmp_path):
    # Setup: Create a temporary database and table for testing
    db = SingleDatabase(db_path=tmp_path, db_name=DB_NAME, tables={TABLE_NAME: TABLE_DATAFRAME})
    yield db  # Provide the fixture object


@pytest.fixture
def null_generator(single_database) -> NullGenerator:
    return NullGenerator(single_database)


def test_get_null_cols(null_generator):
    # Test NullGenerator's _get_null_cols method
    null_cols = null_generator._get_null_cols(TABLE_NAME, sample=2)
    # Assert that the returned null_cols contain only columns with NULL values
    assert set(null_cols) == {'id', 'name'}


def test_null_generator(null_generator):
    generated_sql = null_generator.sql_generate(TABLE_NAME)
    assert 'NULL-COUNT' in generated_sql['sql_tags']
    assert 'NOT-NULL-COUNT' in generated_sql['sql_tags']

    with patch.object(null_generator, '_sample_cat_num_cols', return_value=(pd.DataFrame({}), [], [])):
        generated_sql = null_generator.sql_generate(TABLE_NAME)
        assert len(generated_sql['queries']) == 0
