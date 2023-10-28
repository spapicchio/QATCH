import os
import shutil

import pandas as pd
import pytest

from database_reader import SingleDatabase
from qatch.database_reader import MultipleDatabases

# Define test data
DB_PATH = 'test_db'
DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
TABLE_DICT = {'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}
TABLE_DATAFRAME = pd.DataFrame(TABLE_DICT)


# Fixture to create a temporary MultipleDatabases object for testing
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


# Tests for MultipleDatabases class methods
def test_open_db(multiple_databases):
    for i in range(1, 4):
        db_id = f'{DB_NAME}_{i}'
        multiple_databases.open_db(db_id)
        assert db_id in multiple_databases.db_ids2db


def test_open_db_limit(multiple_databases):
    # Open more databases than the limit, ensure FIFO strategy
    for i in range(20):
        db_id = f'{DB_NAME}_{i}'
        SingleDatabase(db_path=DB_PATH, db_name=db_id, tables={TABLE_NAME: TABLE_DATAFRAME})
        multiple_databases.open_db(db_id)
    assert len(multiple_databases.db_ids2db) == 15


def test_get_table(multiple_databases):
    db_id = f'{DB_NAME}_1'
    # Test retrieving the table from the database
    table_data = multiple_databases.get_table(db_id, TABLE_NAME)
    assert table_data.equals(TABLE_DATAFRAME)


def test_get_schema(multiple_databases):
    db_id = f'{DB_NAME}_1'
    # Test retrieving the schema of the table from the database
    schema = multiple_databases.get_schema(db_id, TABLE_NAME)
    expected_schema = pd.read_sql_query(f"PRAGMA table_info({TABLE_NAME})",
                                        multiple_databases[db_id].conn)
    assert schema.equals(expected_schema)


def test_run_query(multiple_databases):
    db_id = f'{DB_NAME}_1'
    # Test running a query on the database
    query = f"SELECT * FROM {TABLE_NAME}"
    result = multiple_databases.run_query(db_id, query)

    expected_result = TABLE_DATAFRAME.values.tolist()
    assert result == expected_result
