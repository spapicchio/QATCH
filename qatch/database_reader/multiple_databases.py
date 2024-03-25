from __future__ import annotations

import os

import pandas as pd

from .single_database import SingleDatabase


class MultipleDatabases:
    """Manages multiple SQLite databases, allowing dynamic creation and access to individual databases.

    Attributes:
        db_path (str): The base path where the database files are stored.
        db_ids2db (dict): A dictionary where the key is the database name, and the value is the SingleDatabase object.
        _max_db_in_memory (int): The maximum number of databases to keep in memory. Default is 15.
    """

    def __init__(self, db_path: str, _max_db_in_memory=15):
        self.db_path = db_path
        self.db_ids2db: dict[str, SingleDatabase] = dict()
        self._max_db_in_memory = _max_db_in_memory

    def get_db_names(self) -> list[str]:
        """Gets the name of the database file from the path.

        Returns:
            list[str]: A list of database file names.
        """
        # return x only if it is a directory
        return [x for x in os.listdir(self.db_path) if os.path.isdir(os.path.join(self.db_path, x))]

    def __contains__(self, other: str) -> bool:
        # Checks if a database with the given name exists in the managed databases.
        return any(other in db_id for db_id in self.db_ids2db)

    def __getitem__(self, key: str) -> SingleDatabase:
        # Allows accessing a specific database by its name
        # if the key does not exist, open the db
        if key not in self.db_ids2db:
            self.open_db(key)
        return self.db_ids2db[key]

    def open_db(self, db_id: str):
        """Opens a database with the given name and stores it in memory.

        Args:
            db_id (str): The name of the database to open.
        """
        if db_id not in self.db_ids2db:
            if len(self.db_ids2db) >= self._max_db_in_memory:
                # FIFO strategy if max reached
                self.db_ids2db.popitem()
            self.db_ids2db[db_id] = SingleDatabase(self.db_path, db_name=db_id)

    def get_table(self, db_id: str, tbl_name: str) -> pd.DataFrame:
        """Retrieves a specified table from the database as a Pandas DataFrame.

        Args:
            db_id (str): The name of the database.
            tbl_name (str): The name of the table to retrieve from the database.

        Returns:
            pd.DataFrame: A Pandas DataFrame representing the specified table from the database.
        """
        return self[db_id].get_table_given(tbl_name)

    def get_schema(self, db_id: str, tbl_name: str) -> pd.DataFrame:
        """Retrieves the schema of a specified table from the database.

        Args:
            db_id (str): The name of the database.
            tbl_name (str): The name of the table to retrieve the schema from.

        Returns:
            pd.DataFrame: A Pandas DataFrame representing the schema of the specified table.
        """
        return self[db_id].get_schema_given(tbl_name)

    def get_all_table_schema_given(self, db_id: str) -> dict[str, pd.DataFrame]:
        """Retrieves all the table schema from the database.

        Args:
            db_id (str): The name of the database.

        Returns:
            pd.DataFrame: A Pandas DataFrame representing the schema of the specified table.
        """

        return self[db_id].table_schemas

    def run_multiple_queries(self, db_id: str, queries: list) -> list[list]:
        """Executes multiple queries on the specified database and returns the results.

          Args:
              db_id (str): The name of the database to execute the query on.
              queries (list): The list of SQL queries to be executed on the database.

          Returns:
              list[list]: A list containing the query results.
        """
        db = self[db_id]
        queries_result = map(db.run_query, queries)
        queries_result = list(map(lambda x: [list(item) for item in x], queries_result))
        return queries_result

    def run_query(self, db_id: str, query: str) -> list | None:
        """Executes an SQL query on the specified database and returns the results.

        Args:
            db_id (str): The name of the database to execute the query on.
            query (str): The SQL query to be executed on the database.

        Returns:
            list | None: A list containing the query results.
        """
        ans_query = self[db_id].run_query(query)
        return [list(x) for x in ans_query]

    def close(self):
        """Closes all the opened databases and clears the memory."""
        [db.close_connection() for db in self.db_ids2db.values()]
        self.db_ids2db = dict()
