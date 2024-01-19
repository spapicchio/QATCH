import os
import sqlite3

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
    db.close_connection()


# Tests for SingleDatabase class methods
def test_get_table_from_name(single_database):
    table_data = single_database.get_table_given(TABLE_NAME)
    assert table_data.equals(TABLE_DATAFRAME)


def test_get_schema_given(single_database):
    schema = single_database.get_schema_given(TABLE_NAME)
    expected_schema = pd.read_sql_query(f"PRAGMA table_info({TABLE_NAME})", single_database.conn)
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
    db = SingleDatabase(db_path=tmp_path, db_name=DB_NAME, tables={TABLE_NAME: TABLE_DATAFRAME})
    assert TABLE_NAME in db.table_names
    # Teardown: Close the connection and remove the temporary database directory
    db.close_connection()


def test_create_table_in_db():
    name = 'table_1'
    table = pd.DataFrame({'Customer_ID': [1, 2, 3, 4, 5],
                          'Product_ID': [1, 2, 3, 4, 5],
                          'Name': list('simon')})
    primary_key2table = {'Customer_ID': 'table_1', 'Product_ID': 'table_2'}
    create_table_string = SingleDatabase._create_table_in_db(name, table, primary_key2table)
    expected = """CREATE TABLE `table_1`( "Customer_ID" INTEGER, "Product_ID" INTEGER, "Name" TEXT, PRIMARY KEY ("Customer_ID"), FOREIGN KEY (`Product_ID`) REFERENCES `table_2`(`Product_ID`) );"""
    assert expected.strip() == create_table_string.strip()


def test_set_tables_in_db(tmp_path):
    table_1 = pd.DataFrame({'Customer_ID': [1, 2, 3, 4, 5],
                            'Product_ID': [1, 2, 3, 4, 5],
                            'Name': list('simon')})
    table_2 = pd.DataFrame({'Product_ID': [1, 2, 3, 4, 5],
                            'Customer_ID': [1, 2, 3, 4, 5],
                            'Name': list('papic')})
    table2primary_key = {'table_1': 'Customer_ID', 'table_2': 'Product_ID'}

    tables = {'table_1': table_1, 'table_2': table_2}
    conn = sqlite3.connect(os.path.join(tmp_path, f'{DB_NAME}.sqlite'))
    SingleDatabase._set_tables_in_db(tables, conn, table2primary_key)
    table_1_keys = conn.execute("PRAGMA foreign_key_list('{}') ".format('table_1')).fetchall()[0]
    table_2_keys = conn.execute("PRAGMA foreign_key_list('{}') ".format('table_2')).fetchall()[0]
    assert table_1_keys[2] == 'table_2'
    assert table_1_keys[3] == table_1_keys[4] == 'Product_ID'
    assert table_2_keys[3] == table_2_keys[4] == 'Customer_ID'
    assert table_2_keys[2] == 'table_1'
