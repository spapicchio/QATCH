from __future__ import annotations

import contextlib
from itertools import chain

import pandas as pd
from func_timeout import func_set_timeout
from sqlalchemy import create_engine, MetaData, text, Table, String, Numeric, Integer

from .base_connector import BaseConnector, ConnectorTable, ConnectorTableColumn
from .utils import utils_convert_df_in_sql_code


def _convert_sqlalchemy_type_to_string(type_):
    """Convert the SQLAlchemy type to code specific type string"""
    if isinstance(type_, String):
        return 'categorical'
    elif isinstance(type_, (Numeric, Integer)):
        return 'numerical'
    else:
        return None


class SqliteConnector(BaseConnector):
    """
    This class creates the connection between a SQLite database and the Code.

    In case the database does not exist, this class takes a dictionary of tables in which keys are table names
     and values are Pandas DataFrames representing the tables,
    and sets these tables in the SQLite database represented by the `conn` object.
    The optional `table2primary_key` argument can be used to set primary keys for some or all tables.
    If not provided, all tables are created without primary keys.
    If the table contains an attribute with the same name of a primary key, a foreign key relationship is created.

    Note:
        - If a table is named as 'table', the method will replace its name with 'my_table'.

    Args:
        relative_db_path (str): A string representing the relative path to the SQLite database.
        db_name (str): A string representing the name of the database.
        tables (dict[str, pd.DataFrame] | None): An optional dictionary where keys are strings representing table names and values
         are pandas DataFrames, or None.
        table2primary_key (dict[str, pd.DataFrame] | None): An optional dictionary where keys are strings representing table names
         and values are strings representing primary keys, or None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self,
                 relative_db_path: str,
                 db_name: str,
                 tables: dict[str, pd.DataFrame] | None = None,
                 table2primary_key: dict[str, str] | None = None,
                 *args, **kwargs):
        super().__init__(relative_db_path, db_name, *args, **kwargs)
        # Create the engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.metadata = MetaData()
        self.metadata.reflect(self.engine)
        # metadata contains in `tables` a dictionary of {tbl_name: Table}
        if self.metadata.tables and tables:
            raise ValueError('The provided database is not empty, but tables provided')
        elif not self.metadata.tables and not tables:
            raise ValueError('The database is  empty and not tables provided')
        elif not self.metadata.tables and tables:
            self._set_tables_in_db(tables, table2primary_key)
            self.metadata.reflect(self.engine)

    @contextlib.contextmanager
    def connection(self):
        with self.engine.connect() as con:
            yield con

    def load_tables_from_database(self, *args, **kwargs) -> dict[str, ConnectorTable]:
        """
        Loads all tables from the database and creates related ConnectorTable objects for each table.

        The function performs the following steps:
        1. For each table in the database, the function creates a ConnectorTable object.
        2. Next, any necessary foreign key relations are updated for each of these ConnectorTable objects.
        Note:
            - The 2. step cannot be included in the 1. step to avoid infinite loop in referencing key

        Returns:
            A dictionary mapping each table name to its corresponding ConnectorTable object.
        """
        tbl_name2table = {tbl_name: self._create_connector_table_from(tbl_name) for tbl_name in self.metadata.tables}
        tbl_name2table = self._update_foreign_key(tbl_name2table)
        return tbl_name2table

    @func_set_timeout(240)
    def run_query(self, query: str) -> list[list]:
        """
        Executes a query on SQLite database and returns the result as a list of lists.

        Note:
            - This function can't run multiple statements in one go due its use of SQLAlchemy's text-based SQL tool
            which doesn't support multiple SQL statements.
            - The connection to the database auto-closes due to the use of the context manager.
            Hence, the user does not need to explicitly close the connection.
            - This function should not be used with queries that don't return a table of results,
            such as UPDATE or DELETE statements, it is designed for SELECT queries only.
            - If the query execution requires more than 30 seconds, the function will raise FunctionTimedOut

        Args:
            query (str): SQL query string to be executed on the SQLite database.

        Returns:
            list[list]: Returns result of the query in form of list of lists where each member list represents
            a row from the result.
        """
        with self.connection() as con:
            result = con.execute(text(query))
            result = [list(row) for row in result]
        return result

    def _sample_data_from_col(self, col_name, type_, tbl_name):
        """
        Fetches and returns a sample of data from the specified database column.

        This method runs a SQL query to fetch distinct/categorical or numerical data from a specified table and
        column. The limit for fetched data is set to 5 records. The result is converted to a list and returned.

        Args:
            col_name (str): Name of the column to fetch data from.
            type_ (str): Type of the data to fetch. This can be 'categorical' or 'numerical'.
            tbl_name (str): Name of the table where the column resides.

        Returns:
            list: A list containing the fetched sample data.

        Note:
            - 'categorical' type will trigger the SQL query to get distinct data from column
            - Any other type will get top 5 records from that column.
        """

        if type_ == 'categorical':
            result = self.run_query(f"""SELECT DISTINCT `{col_name}` FROM `{tbl_name}` LIMIT 5""")
        else:
            result = self.run_query(f"""SELECT `{col_name}` FROM `{tbl_name}` LIMIT 5""")
        return list(chain.from_iterable(result))

    def _get_columns_metadata_from(self, tbl: Table) -> dict[str, ConnectorTableColumn]:
        """
        Retrieves the metadata of all columns from a given table.

        This method iterates through all the columns of the provided table and builds a dictionary
        where keys are the column names and values are ConnectorTableColumn objects. These objects hold
        the column name, type, and sample data. The column type is determined by converting the
        SQLAlchemy type to string. If the conversion returns None, the column is skipped.

        Note:
            - Columns with an unknown type (where type conversion fails) will not be included in the returned dictionary.
            - Sample data for the columns are obtained by calling self._sample_data_from_col().
            - This method does not perform any type checking on the provided table. It assumes the table contains columns.
            - Can raise exceptions if called with invalid arguments or if an error occurs while handling the table.

        Args:
            tbl (Table): Table object from which metadata is to be collected.

        Returns:
            dict[str, ConnectorTableColumn] : A dictionary with column names as keys and ConnectorTableColumn objects as values.
        """
        columns = tbl.columns._all_columns
        output_dict = dict()
        for col in columns:
            type_string = _convert_sqlalchemy_type_to_string(col.type)
            if not type_string:
                continue

            column = ConnectorTableColumn(
                column_name=col.name,
                column_type=type_string,
                sample_data=self._sample_data_from_col(col.name, type_string, tbl.name)
            )
            output_dict[col.name] = column
        return output_dict

    def _create_connector_table_from(self, tbl_name: str) -> ConnectorTable:
        """
        Instantiate and return a ConnectorTable object based on the table name provided.

        This method retrieves the metadata of the requested table from the database and constructs a ConnectorTable object.

        Note:
        - This method does not provide foreign keys for the ConnectorTable instance. Those will be updated later in
         `_update_foreign_key` method, after all ConnectorTable objects have been instantiated,
        in order to prevent cyclical references between table objects as foreign keys may reference other tables.

        Args:
            tbl_name (str): The name of the table to create the ConnectorTable from.

        Returns:
            An instance of the ConnectorTable representing the specified table.
        """

        tbl = self.metadata.tables[tbl_name]
        tbl_col2metadata = self._get_columns_metadata_from(tbl)
        return ConnectorTable(
            db_path=self.db_path,
            db_name=self.db_name,
            tbl_name=tbl_name,
            tbl_col2metadata=tbl_col2metadata,
            cat_col2metadata={col_name: metadata for col_name, metadata in tbl_col2metadata.items()
                              if metadata.column_type == 'categorical'},
            num_col2metadata={col_name: metadata for col_name, metadata in tbl_col2metadata.items()
                              if metadata.column_type == 'numerical'},
            primary_key=self._extract_primary_key(tbl),
            foreign_keys=[],
        )

    def _set_tables_in_db(self,
                          tables: dict[str, pd.DataFrame] | None,
                          table2primary_key: dict[str, str] | None):
        """
        Method to set the tables in the SQLite database represented by `self.engine`.
        If `tables` or `table2primary_key` is None, an error will be raised.
        If a table is named as 'table', the name will be replaced with 'my_table'.
        If `table2primary_key` is provided, foreign keys will be set according to it.

        Args:
            tables (dict[str, pd.DataFrame] | None): Dictionary containing tables to set into the database.
                Keys are table names and values are corresponding pandas DataFrames.

            table2primary_key (dict[str, str] | None): Dictionary mapping table names to their primary keys.

        Note:
            - If a table is named as 'table', method will change its name to 'my_table' automatically.
            - Primary keys and foreign keys will be set according to `table2primary_key` if provided,
            otherwise tables will be created without primary keys.
        """

        for name, table in tables.items():
            if name == 'table':
                name = 'my_table'
            if not table2primary_key:
                table.to_sql(name, self.engine, if_exists='replace', index=False)
            else:
                create_table_string = utils_convert_df_in_sql_code(name, table, table2primary_key)
                with self.connection() as con:
                    con.execute(text(create_table_string))
                table.to_sql(name, self.engine, if_exists='append', index=False)

    def _extract_primary_key(self, tbl: Table) -> ConnectorTableColumn | None:
        """
        Extracts the primary key from the given table and represents it as an instance of the ConnectorTableColumn class.
        Args:
            tbl (Table): A SQLAlchemy Table object representing the table to extract the primary key from.

        Returns:
            ConnectorTableColumn or None: A list of ConnectorTableColumn instances representing the primary key of
            the table, or None if the table does not have a primary key.

        Note:
            - The type of each column in the primary key is converted to a string.
            - Sample data is fetched from each column in the primary key.
            - If the table does not have a primary key, the function returns None.
        """

        cols = tbl.primary_key.columns
        primary_keys = []
        for col in cols:
            type_string = _convert_sqlalchemy_type_to_string(col.type)
            key = ConnectorTableColumn(
                column_name=col.name,
                column_type=type_string,
                sample_data=self._sample_data_from_col(col.name, type_string, tbl_name=tbl.name)
            )
            primary_keys.append(key)
        return primary_keys if primary_keys else None

    def _update_foreign_key(self, tbl_name2table: dict[str, ConnectorTableColumn]) -> dict[str, ConnectorTableColumn]:
        """
        Updates the foreign key metadata for each table.

        This method iterates over each table and its corresponding metadata. For each table, it goes over
        each foreign key and repackages the key's information into a user-friendly dictionary structure.
        The repackaged foreign key data is then stored back into the original metadata.

        Note:
            - The method modifies the input dictionary directly, and also returns it.
            - The method assumes that the metadata contained in tbl_name2table is up-to-date.

        Args:
            tbl_name2table (dict[str, ConnectorTableColumn]): A dictionary that maps table names to ConnectorTableColumn
                objects, each representing the metadata for a table.

        Returns:
            dict[str, ConnectorTableColumn]: The input dictionary with updated foreign key metadata inserted into
                the ConnectorTableColumn objects.
        """

        for tbl_name, tbl in tbl_name2table.items():
            tbl_sql_alchemy = self.metadata.tables[tbl_name]
            new_keys = []
            for foreign_key in tbl_sql_alchemy.foreign_keys:
                new_key = {
                    'parent_column': foreign_key.parent.name,
                    'child_column': foreign_key.target_fullname.split('.')[1],
                    'child_table': tbl_name2table[foreign_key.target_fullname.split('.')[0]]
                }
                new_keys.append(new_key)

            tbl.foreign_keys = new_keys
        return tbl_name2table
