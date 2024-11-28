import contextlib
from itertools import chain
from typing import Generator

from sqlalchemy import create_engine, MetaData, text, Table, String, Numeric, Integer

from qatch_v_2.connectors.connector import Connector, ConnectorTable, ConnectorTableColumn


class SqliteConnector(Connector):
    def __init__(self, relative_db_path: str, db_name: str, *args, **kwargs):
        super().__init__(relative_db_path, db_name, *args, **kwargs)
        # Create the engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.metadata = MetaData()
        # metadata contains in `tables` a dictionary of tbl_name: Table
        # Each Table
        self.metadata.reflect(self.engine)

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

    @contextlib.contextmanager
    def connection(self):
        with self.engine.connect() as con:
            yield con

    def run_query(self, query: str) -> list[list]:
        with self.connection() as con:
            result = con.execute(text(query))

        result = [list(row) for row in result]
        return result
