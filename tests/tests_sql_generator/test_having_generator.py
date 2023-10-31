import shutil
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from qatch.database_reader import SingleDatabase
from qatch.sql_generator import HavingGenerator

DB_PATH = 'test_db'
DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
NUM_SAMPLES = 10_000


def generate_synthetic_data():
    # Generate random data for each column
    col1 = np.random.normal(1, 10, NUM_SAMPLES)
    col2 = np.random.choice(['A', 'B', 'C'], NUM_SAMPLES)
    col3 = np.random.poisson(10, NUM_SAMPLES)
    col4 = np.random.choice(['Red', 'Blue', 'Green'], NUM_SAMPLES)
    col5 = np.random.choice(['Small', 'Medium', 'Large'], NUM_SAMPLES)
    df = pd.DataFrame({
        'col1': col1,
        'col2': col2,
        'col3': col3,
        'col4': col4,
        'col5': col5,
    })
    return df


TABLE_DATAFRAME = generate_synthetic_data()


# Fixture to create a temporary SingleDatabase object for testing
@pytest.fixture
def single_database():
    # Setup: Create a temporary database and table for testing
    db = SingleDatabase(db_path=DB_PATH, db_name=DB_NAME, tables={TABLE_NAME:
                                                                      TABLE_DATAFRAME})
    yield db  # Provide the fixture object
    # Teardown: Clean up the temporary database and tables after testing
    db.close_connection()
    shutil.rmtree(DB_PATH)


@pytest.fixture
def having_generator(single_database) -> HavingGenerator:
    return HavingGenerator(single_database)


def test_get_average_of_count_cat_col(having_generator):
    cat_col = 'col2'
    start_time = time.time()
    target = int(TABLE_DATAFRAME.groupby(cat_col).count().mean().values[0])
    pd_time = time.time() - start_time
    start_time = time.time()
    avg_cat_col = having_generator._get_average_of_count_cat_col(TABLE_NAME, cat_col)
    qatch_time = time.time() - start_time
    print()
    print(f'Qatch time: {qatch_time}')
    print(f'Pandas time: {pd_time}')


def test_having_generator(having_generator):
    # numerical and categorical attributes present
    with patch.object(having_generator, '_sample_cat_num_cols',
                      return_value=([], ['cat_col_1', 'cat_col_2'], ['num_col_1', 'num_col_2'])):
        generated_sql = having_generator.sql_generate(TABLE_NAME)
        assert 'HAVING-COUNT-EQ' in generated_sql['sql_tags']
        assert 'HAVING-COUNT-LS' in generated_sql['sql_tags']
        assert 'HAVING-COUNT-GR' in generated_sql['sql_tags']
        assert 'HAVING-AGG-AVG-GR' in generated_sql['sql_tags']
        assert 'HAVING-AGG-AVG-LS' in generated_sql['sql_tags']
        assert 'HAVING-AGG-SUM-GR' in generated_sql['sql_tags']
        assert 'HAVING-AGG-SUM-LS' in generated_sql['sql_tags']

        # 3 count for each cat col -> 3 * 2 = 6
        # 4 agg for each comb num_col cat_col -> 4 * 4 = 16
        assert len(generated_sql['queries']) == 22
    # numerical attributes present
    with patch.object(having_generator, '_sample_cat_num_cols',
                      return_value=([], [], ['num_col_1', 'num_col_2'])):
        generated_sql = having_generator.sql_generate(TABLE_NAME)
        # at least one column attribute is required (no groupby otherwise)
        assert len(generated_sql['sql_tags']) == 0
