from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class DistinctGenerator(AbstractSqlGenerator):
    """
    A class for generating DISTINCT SQL queries and corresponding questions based on
     categorical columns of a database table.

    :ivar SingleDatabase database: The SingleDatabase object representing the database to generate queries from.
    :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, and questions.
        Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)
        self.empty_sql_generated()

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Abstract method to generate SQL tags, queries, and questions based on the specified table.
        :param str table_name: The name of the table in the database.
        :return: A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
        """
        _, cat_cols, _ = self._sample_cat_num_cols(table_name)
        self._distinct_single_col(table_name, cat_cols)
        self._distinct_mult_col(table_name, cat_cols)
        return self.sql_generated

    def _distinct_single_col(self, table_name: str, cat_columns: list):
        """
        Generates DISTINCT SQL queries and questions for individual categorical columns.
        :param table_name: The name of the table in the database.
        :param cat_columns: List of categorical column names.
        """

        queries = [f'SELECT DISTINCT "{col}" FROM "{table_name}"'
                   for col in cat_columns]

        questions = [f'Show the different "{col}" in the table {table_name}'
                     for col in cat_columns]

        sql_tags = ['DISTINCT-SINGLE'] * len(queries)
        self.append_sql_generated(sql_tags=sql_tags, queries=queries, questions=questions)

    def _distinct_mult_col(self, table_name: str, cat_columns: list):
        """
        Generates DISTINCT SQL queries and questions for combinations of multiple categorical columns.
        :param table_name: The name of the table in the database.
        :param cat_columns: List of categorical column names.
        """
        combinations = self._comb_random(cat_columns)
        queries = [f'SELECT DISTINCT {self._get_col_comb_str(comb)} FROM "{table_name}"'
                   for comb in combinations]

        questions = [f'Show the different {self._get_col_comb_str(comb)} in the table "{table_name}"'
                     for comb in combinations]

        sql_tags = ['DISTINCT-MULT'] * len(queries)
        self.append_sql_generated(sql_tags=sql_tags, queries=queries, questions=questions)
