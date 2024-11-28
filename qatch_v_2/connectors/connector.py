from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Literal, Optional, Generator

import pandas as pd
from pydantic import BaseModel, Field


class ConnectorTableColumn(BaseModel):
    column_name: str
    column_type: Literal['categorical', 'numerical']
    sample_data: list

    def __hash__(self):
        return hash(self.__dict__['column_name'])


class ConnectorTable(BaseModel):
    db_path: str
    db_name: str
    tbl_name: str
    tbl_col2metadata: dict[str, ConnectorTableColumn]
    cat_col2metadata: dict[str, ConnectorTableColumn]
    num_col2metadata: dict[str, ConnectorTableColumn]


class Connector(ABC):
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
    def load_tables_from_database(self, *args, **kwargs) -> Generator[ConnectorTable, None, None]:
        """Load the table from any possible database.
        This method is a generator, yielding table inputs one-by-one.
        """
        raise NotImplementedError

    @abstractmethod
    def run_query(self, query: str) -> list[list]:
        """Run the query on the database"""
        raise NotImplementedError
