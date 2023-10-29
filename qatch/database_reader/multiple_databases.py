import os

import pandas as pd

from .single_database import SingleDatabase


class MultipleDatabases:
    """
    Manages multiple SQLite databases, allowing dynamic creation
    and access to individual databases.

    :ivar str db_path: The base path where the database files are stored.
    :ivar dict db_ids2db: the key is the database name, the value is the SingleDatabase object.
    :ivar int _max_db_in_memory: The maximum number of databases to keep in memory. Default is 15.
    """

    def __init__(self, db_path: str, _max_db_in_memory=15):
        """
        Initializes the MultipleDatabases object.
        :param str db_path: The base path where the database files will be stored.
        :param int _max_db_in_memory: The maximum number of databases to keep in memory. Default is 15.
        """
        self.db_path = db_path
        self.db_ids2db: dict[str, SingleDatabase] = dict()
        self._max_db_in_memory = _max_db_in_memory

    @staticmethod
    def read_database_name(db_path: str) -> list[str]:
        """
        Gets the name of the database file from the path.
        :param str db_path: The path of the database file.
        :return: A list of database file names.
        """
        return [x for x in os.listdir(db_path)]

    def __contains__(self, other: str) -> bool:
        """
        Checks if a database with the given name exists in the managed databases.
        :param str other: The name of the database to check.
        :return: True if the database exists, False otherwise.
        """
        return any(other in db_id for db_id in self.db_ids2db)

    def __getitem__(self, key) -> SingleDatabase:
        """
        Allows accessing a specific database by its name.
        :param str key: The name of the database to access.
        :return: The SingleDatabase object corresponding to the given database name.
        """
        if key not in self.db_ids2db:
            self.open_db(key)
        return self.db_ids2db[key]

    def open_db(self, db_id: str):
        """
        Opens a database with the given name and stores it in memory.
        :param str db_id: The name of the database to open.
        """
        if db_id not in self.db_ids2db:
            if len(self.db_ids2db) >= self._max_db_in_memory:
                # FIFO strategy if max reached
                self.db_ids2db.popitem()
            self.db_ids2db[db_id] = SingleDatabase(self.db_path, db_name=db_id)

    def get_table(self, db_id, tbl_name) -> pd.DataFrame:
        """
        Retrieves a specified table from the database as a Pandas DataFrame.
        :param str db_id: The name of the database.
        :param str tbl_name: The name of the table to retrieve from the database.
        :return: A Pandas DataFrame representing the specified table from the database.
        """
        return self[db_id].get_table_given(tbl_name)

    def get_schema(self, db_id, tbl_name) -> pd.DataFrame:
        """
        Retrieves the schema of a specified table from the database.
        :param str db_id: The name of the database.
        :param str tbl_name: The name of the table to retrieve the schema from.
        :return: A Pandas DataFrame representing the schema of the specified table.
        """
        return self[db_id].get_schema_given(tbl_name)

    def run_query(self, db_id: str, query: str) -> list:
        """
        Executes an SQL query on the specified database and returns the results.
        :param str db_id: The name of the database to execute the query on.
        :param str query: The SQL query to be executed on the database.
        :return: A list containing the query results.
        """
        ans_query = self[db_id].run_query(query)
        return [list(x) for x in ans_query]

    def close(self):
        """Closes all the opened databases and clears the memory."""
        [db.close_connection() for db in self.db_ids2db.values()]
        self.db_ids2db = dict()
