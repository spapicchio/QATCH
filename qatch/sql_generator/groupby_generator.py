from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class GroupByGenerator(AbstractSqlGenerator):
    """
      A class for generating SQL queries and corresponding questions based on group-by operations
      performed on categorical and numerical columns of a database table.

      :ivar SingleDatabase database: The SingleDatabase object representing the database to generate queries from.
      :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, and questions.
          Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
      """

    def __init__(self, database: SingleDatabase, seed=2023):
        super().__init__(database, seed)

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generates Group By queries and corresponding questions for both categorical and numerical columns.

        :param str table_name: The name of the table in the database.
        :return: A dictionary containing generated SQL tags, queries, questions, and results.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
        """
        self.empty_sql_generated()
        df, cat_cols, num_cols = self._sample_cat_num_cols(table_name)
        self._build_group_by_no_agg(table_name, cat_cols)
        self._build_group_by_with_count(table_name, cat_cols)
        self._build_group_by_with_agg(table_name)
        return self.sql_generated

    def _build_group_by_no_agg(self, table_name: str, cat_cols: list):
        """
        Generate group-by SQL queries and questions without aggregation
        for random combinations of categorical columns.
        The query is the same of Distinct

        :param table_name: The name of the table in the database.
        :param cat_cols: List of categorical columns.
        """
        random_combinations = self._comb_random(cat_cols)

        questions = [f'Show all {self._get_col_comb_str(comb)}' \
                     f' in the table "{table_name}" for each {self._get_col_comb_str(comb)}'
                     for comb in random_combinations]

        queries = [f'SELECT {self._get_col_comb_str(comb)} FROM ' \
                   f'"{table_name}" GROUP BY {self._get_col_comb_str(comb)}'
                   for comb in random_combinations]

        sql_tags = ['GROUPBY-NO-AGGR'] * len(queries)

        self.append_sql_generated(sql_tags=sql_tags, queries=queries,
                                  questions=questions)

    def _build_group_by_with_count(self, table_name: str, cat_cols: list):
        """
        Generate group-by SQL queries and questions with count aggregation for categorical columns.

        :param table_name: The name of the table in the database.
        :param cat_cols: List of categorical columns.
        """
        questions = [f'For each "{col}", count the number of rows in table "{table_name}"'
                     for col in cat_cols]
        queries = [f'SELECT "{col}", COUNT(*) FROM "{table_name}" GROUP BY "{col}"'
                   for col in cat_cols]
        sql_tags = ['GROUPBY-COUNT'] * len(queries)

        self.append_sql_generated(sql_tags=sql_tags, queries=queries,
                                  questions=questions)

    def _build_group_by_with_agg(self, table_name):
        """
        Generate group-by SQL queries and questions with aggregation for numerical columns.

        :param str table_name: The name of the table in the database.
        """
        # with sample == 2 we get 4 tests for each aggregation -> 4*4 = 16 tests
        # with sample == 3 we get 9 tests for each aggregation -> 9*4 = 36 tests
        _, cat_cols, num_cols = self._sample_cat_num_cols(table_name, sample=2)
        for agg in ['min', 'max', 'avg', 'sum']:
            questions = [f'For each "{c_col}", find the {agg} of "{n_col}" in table "{table_name}"'
                         for c_col in cat_cols
                         for n_col in num_cols]

            queries = [f'SELECT "{c_col}", {agg.upper()}("{n_col}") FROM "{table_name}" GROUP BY "{c_col}"'
                       for c_col in cat_cols
                       for n_col in num_cols]

            sql_tags = [f'GROUPBY-AGG-{agg.upper()}'] * len(queries)

            self.append_sql_generated(sql_tags=sql_tags, queries=queries,
                                      questions=questions)
