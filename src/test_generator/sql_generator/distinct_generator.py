import random

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class DistinctGenerator(AbstractSqlGenerator):

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str):
        cat_columns = self._get_categorical_col(table_name)

        output_single = self._generate_distinct_single_col(table_name, cat_columns)
        output_mul = self._generate_distinct_mult_col(table_name, cat_columns)

        sql_tags = output_single[0] + output_mul[0]
        queries = output_single[1] + output_mul[1]
        questions = output_single[2] + output_mul[2]
        results = output_single[3] + output_mul[3]

        return {'sql_tags': sql_tags, 'queries': queries,
                'questions': questions, 'results': results}

    def _get_categorical_col(self, table_name: str) -> list[str]:
        _, cat_cols, _ = self._get_df_cat_num_cols(table_name)
        return cat_cols

    def _generate_distinct_single_col(self, table_name, cat_columns):
        queries = [
            f'SELECT DISTINCT "{comb}" FROM "{table_name}"'
            for comb in cat_columns
        ]

        questions = [
            f'Show the different {comb} in the table {table_name}'
            for comb in cat_columns
        ]

        # run the query and get the results
        results = [self.database.run_query(query) for query in queries]
        sql_tags = ['DISTINCT-SINGLE'] * len(queries)
        return sql_tags, queries, questions, results

    def _generate_distinct_mult_col(self, table_name, cat_columns):
        def _get_col_comb(comb):
            return ", ".join([f'"{str(c)}"' for c in comb])

        combinations = self._comb_random(cat_columns)
        queries = [
            f'SELECT DISTINCT {_get_col_comb(comb)} FROM "{table_name}"'
            for comb in combinations
        ]

        questions = [
            f'Show the different {_get_col_comb(comb)} in the table "{table_name}"'
            for comb in combinations
        ]

        # run the query and get the results
        results = [self.database.run_query(query) for query in queries]
        sql_tags = ['DISTINCT-MULT'] * len(queries)
        return sql_tags, queries, questions, results

