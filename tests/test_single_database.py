import os
import shutil
import sqlite3

import pandas as pd
import pytest

from qatch.database_reader import SingleDatabase

# Define test data
DB_PATH = 'test_db'
DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
TABLE_DICT = {'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}
TABLE_DATAFRAME = pd.DataFrame(TABLE_DICT)


# Fixture to create a temporary SingleDatabase object for testing
@pytest.fixture
def single_database():
    # Setup: Create a temporary database and table for testing
    db = SingleDatabase(db_path=DB_PATH, db_name=DB_NAME, tables={TABLE_NAME: TABLE_DATAFRAME})
    yield db  # Provide the fixture object
    # Teardown: Clean up the temporary database and tables after testing
    db.close_connection()
    shutil.rmtree(DB_PATH)


# Tests for SingleDatabase class methods
def test_get_table_from_name(single_database):
    table_data = single_database.get_table_from_name(TABLE_NAME)
    assert table_data.equals(TABLE_DATAFRAME)


def test_get_schema_given(single_database):
    schema = single_database.get_schema_given(TABLE_NAME)
    expected_schema = pd.read_sql_query(f"PRAGMA table_info({TABLE_NAME})", single_database.conn)
    assert schema.equals(expected_schema)


def test_run_query(single_database):
    query = f"SELECT * FROM {TABLE_NAME}"
    result = single_database.run_query(query)
    expected_result = [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]
    assert result == expected_result


def test_nonexistent_table(single_database):
    with pytest.raises(pd.errors.DatabaseError):
        single_database.get_table_from_name('nonexistent_table')


def test_nonexistent_schema(single_database):
    with pytest.raises(KeyError):
        single_database.get_schema_given('nonexistent_table')


def test_existing_db_without_tables():
    # Setup: Create an empty database without providing tables
    db_path = os.path.join(DB_PATH, DB_NAME)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        conn = sqlite3.connect(os.path.join(db_path, f'{DB_NAME}.sqlite'))
        conn.close()

    # Test: Attempt to initialize SingleDatabase without tables
    with pytest.raises(ValueError):
        SingleDatabase(db_path=DB_PATH, db_name=DB_NAME)


def test_existing_db_with_tables():
    # Setup: Create a database with existing tables
    db_path = os.path.join(DB_PATH, DB_NAME)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        conn = sqlite3.connect(os.path.join(DB_PATH, DB_NAME, f'{DB_NAME}.sqlite'))
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE {TABLE_NAME} (id INTEGER, name TEXT)")
        conn.commit()
        conn.close()

    # Test: Initialize SingleDatabase with tables
    db = SingleDatabase(db_path=DB_PATH, db_name=DB_NAME, tables={TABLE_NAME: TABLE_DATAFRAME})
    assert TABLE_NAME in db.table_names
    # Teardown: Close the connection and remove the temporary database directory
    db.close_connection()
    shutil.rmtree(DB_PATH)