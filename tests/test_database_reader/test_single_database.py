import os
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from qatch.database_reader import SingleDatabase

# Define test data
DB_NAME = 'test_database'
TABLE_NAME = 'test_table'
TABLE_DICT = {'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}
TABLE_DATAFRAME = pd.DataFrame(TABLE_DICT)


# Fixture to create a temporary SingleDatabase object for testing
@pytest.fixture
def single_database(tmp_path):
    tmp_path = str(tmp_path)
    # Setup: Create a temporary database and table for testing
    db = SingleDatabase(db_path=tmp_path, db_name=DB_NAME, tables={TABLE_NAME: TABLE_DATAFRAME})
    yield db  # Provide the fixture object
    # Teardown: Clean up the temporary database and tables after testing


def test_init_database(tmp_path):
    tmp_path = str(tmp_path)
    # Initial setup
    table = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}], columns=['a', 'b'])
    tables = {'test_table': table}
    primary_keys = {'test_table': 'a'}

    # Stage 1: the database does not exist -> we create one
    db = SingleDatabase(tmp_path, 'fake_db', tables, primary_keys)
    assert db.table_names == ['test_table']

    # Stage 2: the database does exist -> we do not overwrite
    new_table = pd.DataFrame([{"c": 5, "d": 6}, {"c": 7, "d": 8}], columns=['c', 'd'])
    db = SingleDatabase(tmp_path, 'fake_db', {'new_table': new_table}, None)
    assert db.table_names == ['test_table']

    # Stage 3: Same table of stage 1
    result = db.run_query("SELECT * FROM test_table")
    assert result == [(1, 2), (3, 4)]


# Tests for SingleDatabase class methods
def test_get_table_from_name(single_database):
    table_data = single_database.get_table_given(TABLE_NAME)
    assert table_data.equals(TABLE_DATAFRAME)


def test_get_schema_given(single_database):
    schema = single_database.get_schema_given(TABLE_NAME)
    with single_database.connect_cursor() as cursor:
        conn = cursor.connection
        expected_schema = pd.read_sql_query(f"PRAGMA table_info({TABLE_NAME})", conn)
    assert schema.equals(expected_schema)


def test_run_query(single_database):
    # upper to check whether sqlite is case-sensitive or not
    query = f"SELECT * FROM {TABLE_NAME.upper()}"
    result = single_database.run_query(query)
    expected_result = [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]
    assert result == expected_result


def test_nonexistent_table(single_database):
    with pytest.raises(pd.errors.DatabaseError):
        single_database.get_table_given('nonexistent_table')


def test_nonexistent_schema(single_database):
    with pytest.raises(KeyError):
        single_database.get_schema_given('nonexistent_table')


def test_existing_db_without_tables(tmp_path):
    tmp_path = str(tmp_path)
    # Setup: Create an empty database without providing tables
    db_path = os.path.join(tmp_path, DB_NAME)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        conn = sqlite3.connect(os.path.join(db_path, f'{DB_NAME}.sqlite'))
        conn.close()

    # Test: Attempt to initialize SingleDatabase without tables
    with pytest.raises(ValueError):
        SingleDatabase(db_path=tmp_path, db_name=DB_NAME)


def test_existing_db_with_tables(tmp_path):
    # Setup: Create a database with existing tables
    db_path = os.path.join(tmp_path, DB_NAME)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        conn = sqlite3.connect(os.path.join(tmp_path, DB_NAME, f'{DB_NAME}.sqlite'))
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE {TABLE_NAME} (id INTEGER, name TEXT)")
        conn.commit()
        conn.close()

    # Test: Initialize SingleDatabase with tables
    db = SingleDatabase(db_path=str(tmp_path), db_name=DB_NAME, tables={TABLE_NAME: TABLE_DATAFRAME})
    assert TABLE_NAME in db.table_names


def test_create_table_in_db():
    name = 'table_1'
    table = pd.DataFrame({'Customer_ID': [1, 2, 3, 4, 5],
                          'Product_ID': [1, 2, 3, 4, 5],
                          'Name': list('simon')})
    table2primary_key = {'table_1': 'Customer_ID', 'table_2': 'Product_ID'}
    create_table_string = SingleDatabase._create_table_in_db(name, table, table2primary_key)
    expected = """CREATE TABLE `table_1`( "Customer_ID" INTEGER, "Product_ID" INTEGER, "Name" TEXT, PRIMARY KEY ("Customer_ID"), FOREIGN KEY (`Product_ID`) REFERENCES `table_2`(`Product_ID`) );"""
    assert expected.strip() == create_table_string.strip()


def test_set_tables_in_db(tmp_path):
    # 1. Two sample dataframes are created that represent tables
    table_1 = pd.DataFrame({'Customer_ID': [1, 2, 3, 4, 5],
                            'Product_ID': [1, 2, 3, 4, 5],
                            'Name': list('simon')})
    table_2 = pd.DataFrame({'Product_ID': [1, 2, 3, 4, 5],
                            'Customer_ID': [1, 2, 3, 4, 5],
                            'Name': list('papic')})

    # 2. Define primary key for each table
    table2primary_key = {'table_1': 'Customer_ID', 'table_2': 'Product_ID'}

    # 3. Put the tables into a dictionary
    tables = {'table_1': table_1, 'table_2': table_2}

    # 4. Connect to the SQLite database
    DB_NAME = 'test'

    # 5. Instantiate the SingleDatabase object
    db = SingleDatabase(db_path=tmp_path, db_name=DB_NAME, tables=tables, table2primary_key=table2primary_key)
    conn = sqlite3.connect(db.db_path_sqlite)
    # 6. Check the foreign keys of the tables in the database
    table_1_keys = conn.execute("PRAGMA foreign_key_list('{}') ".format('table_1')).fetchall()[0]
    table_2_keys = conn.execute("PRAGMA foreign_key_list('{}') ".format('table_2')).fetchall()[0]

    # 8. These assertions verify that the tables have been created correctly.
    assert table_1_keys[2] == 'table_2'
    assert table_1_keys[3] == table_1_keys[4] == 'Product_ID'
    assert table_2_keys[3] == table_2_keys[4] == 'Customer_ID'
    assert table_2_keys[2] == 'table_1'


# Behavior Test
@pytest.mark.parametrize("db_path, db_name, cwd, expected", [
    ('.', 'database', '/current_directory', '/current_directory/database'),
    ('./', 'database', '/current_directory', '/current_directory/database'),
    ('', 'database', '/current_directory', '/current_directory/database'),
    ('/path', 'database', '/current_directory', '/path/database')
])
def test__ensure_valid_directory(db_path, db_name, cwd, expected, monkeypatch):
    # Setup
    result = str(Path(expected))

    # Mock dependencies
    def mock_getcwd():
        return cwd

    monkeypatch.setattr(os, 'getcwd', mock_getcwd)
    monkeypatch.setattr(os, 'makedirs', lambda x, exist_ok: None)

    # Exercise
    actual = SingleDatabase._ensure_valid_directory(db_path, db_name)

    # Verify
    assert actual == result
