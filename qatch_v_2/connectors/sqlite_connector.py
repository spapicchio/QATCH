from __future__ import annotations

import contextlib
from itertools import chain
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine, MetaData, text, Table, String, Numeric, Integer

from .connector import Connector, ConnectorTable, ConnectorTableColumn
from .utils import utils_convert_df_in_sql_code


class SqliteConnector(Connector):
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
        # metadata contains in `tables` a dictionary of tbl_name: Table
        # Each Table

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

    def load_tables_from_database(self, *args, **kwargs) -> Generator[ConnectorTable, None, None]:
        for tbl_name, tbl in self.metadata.tables.items():
            tbl_col2metadata = self.get_columns_metadata_from(tbl)
            yield ConnectorTable(
                db_path=self.db_path,
                db_name=self.db_name,
                tbl_name=tbl_name,
                tbl_col2metadata=tbl_col2metadata,
                cat_col2metadata={col_name: metadata for col_name, metadata in tbl_col2metadata.items()
                                  if metadata.column_type == 'categorical'},
                num_col2metadata={col_name: metadata for col_name, metadata in tbl_col2metadata.items()
                                  if metadata.column_type == 'numerical'},

            )

    def run_query(self, query: str) -> list[list]:
        with self.connection() as con:
            result = con.execute(text(query))
        result = [list(row) for row in result]
        return result

    def get_columns_metadata_from(self, tbl: Table) -> dict[str, ConnectorTableColumn]:
        def convert_sqlalchemy_type_to_string(type_):
            if isinstance(type_, String):
                return 'categorical'
            elif isinstance(type_, (Numeric, Integer)):
                return 'numerical'
            else:
                return None

        def sample_data_from_col(col_, type_):
            if type_ == 'categorical':
                result = self.run_query(f"""SELECT DISTINCT `{col_.name}` FROM `{tbl.name}` LIMIT 5""")
            else:
                result = self.run_query(f"""SELECT `{col_.name}` FROM `{tbl.name}` LIMIT 5""")
            return list(chain.from_iterable(result))

        columns = tbl.columns._all_columns
        output_dict = dict()
        for col in columns:
            type_string = convert_sqlalchemy_type_to_string(col.type)
            if not type_string:
                continue

            column = ConnectorTableColumn(
                column_name=col.name,
                column_type=type_string,
                sample_data=sample_data_from_col(col, type_string)
            )
            output_dict[col.name] = column
        return output_dict

    def _set_tables_in_db(self,
                          tables: dict[str, pd.DataFrame] | None,
                          table2primary_key: dict[str, str] | None):
        """
        Sets the tables in the SQLite database represented by the given connection object.

        This method takes a dictionary of tables in which keys are table names and values are Pandas DataFrames
        representing the tables, and sets these tables in the SQLite database represented by the `conn` object.

        The optional `table2primary_key` argument can be used to set primary keys for some or all tables.
        If not provided, all tables are created without primary keys.
        If the table contains an attribute with the same name of a primary key, a foreign key relationship is created.

        Note:
            - If a table is named as 'table', the method will replace its name with 'my_table'.
            - Assume the PKs have all different names. two tables must have different PK names.

        Args:
            tables (Optional[Dict[str, pd.DataFrame]]): A dictionary of tables to set in the SQLite database.
                Keys are table names and values are corresponding Pandas DataFrames.

            table2primary_key (Optional[Dict[str, str]]): A dictionary mapping table names to primary keys.
                For example, if you want to set the primary key of table `A` to be `Key_1`, you should pass
                `table2primary_key={'A': 'Key_1'}`. Default is None.
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
