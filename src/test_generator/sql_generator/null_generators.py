import random

from test_generator.database_reader import SingleDatabase
from test_generator.sql_generator.abstract_sql_generator import AbstractSqlGenerator


class NullGenerator(AbstractSqlGenerator):
    def __init__(self, database: SingleDatabase, seed=2023):
        super().__init__(database, seed)
        self.sql_generated = {'sql_tags': [], 'queries': [], 'questions': [], 'results': []}

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """the group by is performed only with the categorical columns"""
        self.sql_generated = {'sql_tags': [], 'queries': [], 'questions': [], 'results': []}
        self._build_null_count(table_name)
        self._build_null_not_count(table_name)
        # self._build_null_with_no_null_col(table_name)
        return self.sql_generated

    def _build_null_count(self, table_name: str):
        # TODO not NULL in values, pay attention
        null_cols = self._get_null_cols(table_name)
        queries = [f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NULL'
                   for col in null_cols]

        questions = [f'Count the rows where the values of "{col}" are missings in table "{table_name}"'
                     for col in null_cols]

        results = [self.database.run_query(query) for query in queries]

        sql_tags = ['NULL-COUNT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions, results)

    def _build_null_not_count(self, table_name):
        null_cols = self._get_null_cols(table_name)
        queries = [f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NOT NULL'
                   for col in null_cols]

        questions = [f'Count the rows where the values of "{col}" are not missings in table "{table_name}"'
                     for col in null_cols]

        results = [self.database.run_query(query) for query in queries]

        sql_tags = ['NOT-NULL-COUNT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions, results)

    def _build_null_with_no_null_col(self, table_name):
        # TODO error cat_cols sample larger than the number of columns
        _, cat_cols, _ = self._get_df_cat_num_cols(table_name)
        null_cols = self._get_null_cols(table_name)
        cat_cols = random.sample([col for col in cat_cols if col not in null_cols], 2)

        queries = [f'SELECT * FROM "{table_name}" WHERE "{null_col}" IS NULL AND {cat_col} IS NOT NULL'
                   for null_col in null_cols for cat_col in cat_cols]
        questions = [f'Retrieve rows where "{null_col}" has NULL values and ' \
                     f'"{cat_col}" is not NULL in table "{table_name}"'
                     for null_col in null_cols for cat_col in cat_cols]

        results = [self.database.run_query(query) for query in queries]

        sql_tags = ['NULL-NOT-NULL-COLS'] * len(queries)
        self.extend_values_generated(sql_tags, queries, questions, results)

    def _get_null_cols(self, table_name: str, sample=2):
        df, _, _ = self._get_df_cat_num_cols(table_name)
        mask = df.isnull().any()
        cols = list(df.columns[mask])
        return random.sample(cols, sample) if len(cols) >= sample else cols


