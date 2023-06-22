import random

import pandas as pd

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class SelectGenerator(AbstractSqlGenerator):
    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str):
        df = self.database.tables[table_name]

        queries = [f'SELECT * FROM "{table_name}"']
        questions = [f"Show all the rows in the table {table_name}"]
        results = [df.values.tolist()]

        columns = df.columns.tolist()

        comb_cols_add = self._comb_add_columns(columns)
        comb_cols_rand = self._comb_random(columns)
        comb_cols = comb_cols_add + comb_cols_rand

        questions += self._build_questions(comb_cols, table_name)
        queries += self._build_queries(comb_cols, table_name)
        results += self._get_query_results(df, comb_cols)
        sql_tags = ['SELECT-ALL'] + \
                   ['SELECT-ADD-COL'] * len(comb_cols_add) + \
                   ['SELECT-RANDOM-COL'] * len(comb_cols_rand)

        return {'sql_tags': sql_tags, 'queries': queries,
                'questions': questions, 'results': results}

    def _build_questions(self, combinations: list[list[str]], table_name) -> list[str]:
        """fixed question template for SELECT queries
        :param table_name
        """
        return [f'Show all {self._get_col_comb_str(comb)} in the table {table_name}'
                for comb in combinations]

    def _build_queries(self, combinations: list[list[str]], table_name: str) -> list[str]:
        """fixed template for SELECT
        :param table_name:
        """
        return [f'SELECT {self._get_col_comb_str(comb)} FROM "{table_name}"'
                for comb in combinations]

    @staticmethod
    def _get_query_results(df: pd.DataFrame, comb_cols: list[list[str]]
                           ) -> list[list[list[str]]]:
        """get results from queries, faster than executing the query with sqlite"""
        return [df[list(comb)].values.tolist() for comb in comb_cols]

    @staticmethod
    def _comb_add_columns(columns: list[str]) -> list[list[str]]:
        """increasingly add columns to the query"""
        return [columns[:i] for i in range(1, len(columns))]
