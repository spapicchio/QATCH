import pandas as pd

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class WhereGenerator(AbstractSqlGenerator):

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str):
        """
        several WHERE conditions are tested:
            * WHERE column_name = value (only for categorical columns)
            * WHERE column_name != value
            * WHERE column_name > value
            * WHERE column_name < value
            * WHERE column_name >= value
            * WHERE column_name <= value
        """
        cat_cols, num_cols = self._get_cat_num_cols(table_name)
        df = self.database.get_table_from_name(table_name)
        output_cat = self._where_categorical(df, table_name, cat_cols)
        output_num = self._where_numerical(df, table_name, num_cols)

        sql_tags = output_cat[0] + output_num[0]
        queries = output_cat[1] + output_num[1]
        questions = output_cat[2] + output_num[2]

        results = [self.database.run_query(query) for query in queries]

        assert (len(sql_tags) == len(queries) == len(questions) == len(results))
        return {'sql_tags': sql_tags, 'queries': queries,
                'questions': questions, 'results': results}

    @staticmethod
    def _where_categorical(df: pd.DataFrame, tbl_name: str, cat_cols: list[str]
                           ) -> tuple[list, list, list]:
        if len(cat_cols) == 0:
            return [], [], []
        operation2str = {'!=': 'is different from', '=': 'is equal to'}
        min_values = [df[col].min() for col in cat_cols]
        max_values = [df[col].max() for col in cat_cols]
        queries, questions = [], []

        for values in [max_values, min_values]:
            for oper in ['!=', '=']:
                queries += [
                    f'SELECT * FROM "{tbl_name}" WHERE "{col}" {oper} "{value}"'
                    for col, value in zip(cat_cols, values)
                ]

                questions += [
                    f'Show the data of the table {tbl_name} where' \
                    f' {col} {operation2str[oper]} {value}'
                    for col, value in zip(cat_cols, values)
                ]

        sql_tags = ['WHERE-CAT-MAX-VALUES'] * len(max_values) * 2
        sql_tags += ['WHERE-CAT-MIN-VALUES'] * len(min_values) * 2

        return sql_tags, queries, questions

    @staticmethod
    def _where_numerical(df: pd.DataFrame, tbl_name: str, num_cols: list[str]
                         ) -> tuple[list, list, list]:
        if len(num_cols) == 0:
            return [], [], []
        operation2str = {'>': 'is greater than',
                         '<': 'is less than',
                         '>=': 'is greater than or equal to',
                         '<=': 'is less than or equal to', }
        queries, questions = [], []
        max_values = [df[col].max() for col in num_cols]
        min_values = [df[col].min() for col in num_cols]
        mean_values = [df[col].mean() for col in num_cols]

        for values in [max_values, min_values, mean_values]:
            for oper in ['>', '<', '>=', '<=']:
                queries += [
                    f'SELECT * FROM "{tbl_name}" WHERE "{col}" {oper} "{value}"'
                    for col, value in zip(num_cols, values)
                ]

                questions += [
                    f'Show the data of the table {tbl_name} where' \
                    f' {col} {operation2str[oper]} {value}'
                    for col, value in zip(num_cols, values)
                ]
        sql_tags = ['WHERE-NUM-MAX-VALUES'] * len(max_values) * 4
        sql_tags += ['WHERE-NUM-MIN-VALUES'] * len(min_values) * 4
        sql_tags += ['WHERE-NUM-MEAN-VALUES'] * len(mean_values) * 4

        return sql_tags, queries, questions


