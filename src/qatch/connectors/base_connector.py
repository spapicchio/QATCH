from __future__ import annotations

import os
from abc import ABC, abstractmethod

import pandas as pd
from func_timeout import func_set_timeout
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal, TypedDict


class ConnectorTableColumn(BaseModel):
    column_name: str  # The column name in the table
    # The datatype of the column, can be either 'categorical' or 'numerical'
    column_type: Literal['categorical', 'numerical']
    sample_data: list  # A list of sample data values from the column

    def __hash__(self):
        # This method returns a hash value of the column name.
        # It's a unique integer that represents the column,
        # so it can be used in hash-based data structures like sets or dicts
        return hash(self.__dict__['column_name'])


class ForeignKey(TypedDict):
    parent_column: str  # The name of the parent column involved in the foreign key relationship
    child_column: str  # The name of the child column involved in the foreign key relationship
    child_table: ConnectorTable  # The child table object in the foreign key relationship


class Config:
    arbitrary_types_allowed = True


class ConnectorTable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    db_path: str  # The path to the database
    db_name: str  # The name of the database
    tbl_name: str  # The name of the table
    tbl_col2metadata: dict[str, ConnectorTableColumn]  # Mapping of table column names to their metadata.
    cat_col2metadata: dict[str, ConnectorTableColumn]  # Mapping of category column names to their metadata.
    num_col2metadata: dict[str, ConnectorTableColumn]  # Mapping of numeric column names to their metadata.
    primary_key: list[ConnectorTableColumn] | None  # The primary keys of the table as list of ConnectorTableColumn.
    foreign_keys: list[ForeignKey]  # List of foreign keys in the table.


class BaseConnector(ABC):
    """
    This abstract class serves as the base for all connectors that connect the application to a database.

    The BaseConnector initializes the connection with the database and includes abstract methods that need to be implemented
    in all derived connector classes. These methods include `load_tables_from_database` and `run_query`.

    Note:
    The BaseConnector is an abstract base class that cannot be instantiated directly.
    Implement the abstract methods in a derived class.

    Attributes:
        db_path (str): The path to the database file.
        db_name (str): The name of the database.
        tables (dict[str, pd.DataFrame]): A dictionary mapping from table names to pandas DataFrame objects.
                                           Each DataFrame represents a table in the database.
        table2primary_key (dict[str, str]): A dictionary mapping from table names to their primary keys.

    Args:
        relative_db_path (str): A string representing the relative path to the database.
        db_name (str): A string representing the name of the database.
        tables (dict[str, pd.DataFrame] | None): An optional dictionary where keys are strings representing table names and values
         are pandas DataFrames, or None.
        table2primary_key (dict[str, pd.DataFrame] | None): An optional dictionary where keys are strings representing table names
         and values are strings representing primary keys, or None.
        *args : Variable length argument list.
        **kwargs : Arbitrary keyword arguments.
    """

    def __init__(self,
                 relative_db_path: str,
                 db_name: str,
                 tables: dict[str, pd.DataFrame] | None = None,
                 table2primary_key: dict[str, str] | None = None,
                 *args, **kwargs):
        self.db_path = relative_db_path if '.sqlite' in relative_db_path else os.path.join(relative_db_path,
                                                                                           f'{db_name}.sqlite')
        self.db_name = db_name
        self.tables = tables
        self.table2primary_key = table2primary_key

    @abstractmethod
    def load_tables_from_database(self, *args, **kwargs) -> dict[str, ConnectorTable]:
        """
        Load tables from any connected database.

        This method tries to load all the tables available in the database connected by the current connector.
        Each loaded table will get transformed into a `ConnectorTable` object that includes detailed metadata
        about the table, like its columns, primary key, foreign keys, etc.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict[str, ConnectorTable]: A dictionary mapping from table names to `ConnectorTable` objects.
                                       Each `ConnectorTable` object represents a table in the database,
                                       along with its metadata.
        """

        raise NotImplementedError

    @abstractmethod
    @func_set_timeout(30)
    def run_query(self, query: str) -> list[list]:
        """
        Run the query on the database.
        If the execution of the query is gt 30 seconds, the query will raise `FunctionTimedOut`

        Args:
        query (str): The SQL query to be executed.

        Returns:
            list[list]: Returns the results of the query as a list of lists.
                       Each inner list represents a row extracted from the result set of the query"""
        raise NotImplementedError
