import os.path
import sqlite3

import pandas as pd


class SingleDatabase:

    def __init__(self, db_path: str, db_name: str,
                 tables: dict[str, pd.DataFrame] | None = None):
        if not os.path.exists(os.path.join(db_path, db_name)):
            os.makedirs(os.path.join(db_path, db_name))

        # SPIDER database path
        db_path = os.path.join(db_path, f'{db_name}.sqlite')

        if not os.path.exists(db_path) and tables is None:
            raise ValueError(f"Database path does not exist and no tables were provided."
                             f" Path {db_path}")

        # This creates a connection object that represents the database
        self.conn = sqlite3.connect(db_path)
        # A cursor is used to execute SQL commands and retrieve results.
        self.cursor = self.conn.cursor()

        if os.path.exists(db_path) and tables is not None:
            self.cursor.execute('SELECT name from sqlite_master where type= "table"')
            existing_tables = [tbl[0] for tbl in self.cursor.fetchall()]
            self.set_tables_in_db(tables)

        self.db_name = db_name

        self.tables = tables
        self.set_tables_in_db(tables)

    def set_tables_in_db(self, tables: dict[str, pd.DataFrame] | None):
        if tables is not None:
            self.tables = tables
            for name, table in tables.items():
                if name == 'table':
                    name = 'my_table'
                table.to_sql(name, self.conn, if_exists='replace', index=False)

    def get_table_from_name(self, table_name: str) -> pd.DataFrame:
        if self.tables is None:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
        else:
            return self.tables[table_name]

    def run_query(self, query: str) -> list:
        """Run a query on the database and return the result
        only the first _MAX_TABLE_SIZE rows are analyzed"""
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close_connections(self):
        self.conn.close()

    def get_columns_from_table(self, table_name: str) -> list:
        return self.get_table_from_name(table_name).columns.tolist()
