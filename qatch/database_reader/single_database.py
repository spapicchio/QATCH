import logging
import os.path
import sqlite3

import pandas as pd


class SingleDatabase:
    """Provides a simplified interface for interacting with SQLite databases in Python.

    Attributes:
        conn (sqlite3.Connection): A connection object representing the SQLite database.
        cursor (sqlite3.Cursor): A cursor object used to execute SQL commands and retrieve results.
        db_name (str): The name of the database.
        table_schemas (Dict[str, pd.DataFrame]): Table name as key, table schema as value.
            Table schema is a Pandas DataFrame with columns: cid, name, type, notnull, dflt_value, pk.
        table_names (str): The names of the tables in the database.
        db_path (str): Path to the folder which contains the SQLite file.
        db_path_sqlite (str): Path to the SQLite file.
    """

    def __init__(self, db_path: str, db_name: str, tables: dict[str, pd.DataFrame] | None = None):
        """
        Initializes a new SingleDatabase object.

        Args:
            db_path (str): The path where the database file will be stored, or it is already stored.
            db_name (str): The name of the database file (without the file extension).
            tables (Optional[Dict[str, pd.DataFrame]]): A dictionary containing table names as keys and corresponding
                Pandas DataFrames as values. If provided, these tables will be created in the database upon
                initialization. Default is None.

        Raises:
            ValueError: If the specified `db_path` does not exist and no tables are provided.

        """
        if db_path == '' or db_path == '.' or db_path == './':
            db_path = os.getcwd()
        db_path = os.path.join(db_path, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        path_sqlite_file = os.path.join(db_path, f'{db_name}.sqlite')
        # This creates a connection object that represents the database in case it does not exist
        conn = sqlite3.connect(path_sqlite_file)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        # A cursor is used to execute SQL commands and retrieve results.
        cursor = conn.cursor()
        # select all the table names from the database if any
        cursor.execute('SELECT name from sqlite_master where type= "table"')
        existing_tables = [tbl[0] for tbl in cursor.fetchall()]
        if len(existing_tables) == 0:
            if tables is None:
                # CASE 1: no tables provided and database does not exist
                # remove the created file
                conn.close()
                raise ValueError(f"Database path does not exist and no tables were provided "
                                 f"to create the new database in the given path:"
                                 f" {path_sqlite_file}")
            else:
                # CASE 2: tables provided and database does not exist
                logging.info(f"provided tables are stored in {path_sqlite_file}")
                existing_tables = list(tables.keys())
                self._set_tables_in_db(tables, conn)
        else:
            if tables is not None:
                # CASE 3: database already exists and tables are provided
                # log warning tables provided but also database already exists
                logging.warning("tables provided but also database already exists, "
                                "tables in the database will be used")
        self.db_path_sqlite = path_sqlite_file
        self.db_path = db_path
        self.db_name = db_name
        self.conn = conn
        self.cursor = cursor
        self.table_names = existing_tables
        self.table_schemas = {tbl_name: pd.read_sql_query(f'PRAGMA table_info("{tbl_name}")', self.conn)
                              for tbl_name in self.table_names}

    @staticmethod
    def _set_tables_in_db(tables: dict[str, pd.DataFrame] | None, conn: sqlite3.Connection):
        """
        Sets the tables in the database based on the provided dictionary.

        Args:
            tables (Optional[Dict[str, pd.DataFrame]]): A dictionary containing table names as keys and corresponding
                Pandas DataFrames as values.
            conn (sqlite3.Connection): A connection object representing the SQLite database.
        """
        for name, table in tables.items():
            if name == 'table':
                name = 'my_table'
            table.to_sql(name, conn, if_exists='replace', index=False)

    def get_table_given(self, table_name: str) -> pd.DataFrame:
        """
        Retrieves a specified table from the database as a Pandas DataFrame.

        Args:
            table_name (str): The name of the table to retrieve from the database.

        Returns:
            pd.DataFrame: A Pandas DataFrame representing the specified table from the database.
        """
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)

    def get_schema_given(self, table_name: str) -> pd.DataFrame:
        """
        Given the table name, returns the schema of the table.
        Table schema is a Pandas DataFrame with columns: cid, name, type, notnull, dflt_value, pk.

        Args:
            table_name (str): The name of the table to retrieve the schema from the database.

        Returns:
            pd.DataFrame: A Pandas DataFrame representing the schema of the specified table from the database.
        """
        return self.table_schemas[table_name]

    def run_query(self, query: str) -> list[list]:
        """
        Run a query on the database and return the result.

        Args:
            query (str): The SQL query to be executed on the database.
        Returns:
            list[list]: A list of lists representing the result of the SQL query.
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close_connection(self):
        """Closes the connection to the database."""
        self.conn.close()
