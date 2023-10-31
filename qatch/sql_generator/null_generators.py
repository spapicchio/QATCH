import random

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class NullGenerator(AbstractSqlGenerator):
    """
    A class for generating NULL SQL queries and corresponding questions based on a database table.

    :ivar SingleDatabase database: The SingleDatabase object representing the database to generate queries from.
    :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, and questions.
        Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def __init__(self, database: SingleDatabase, seed=2023):
        super().__init__(database, seed)

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generates NULL queries and corresponding questions.

        :param str table_name: The name of the table in the database.
        :return: A dictionary containing generated SQL tags, queries, questions, and results.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
        """
        self.empty_sql_generated()
        null_cols = self._get_null_cols(table_name)
        self._build_null_count(table_name, null_cols)
        self._build_null_not_count(table_name, null_cols)
        # self._build_null_with_no_null_col(table_name)
        return self.sql_generated

    def _build_null_count(self, table_name: str, null_cols: list[str]):
        """
        Build SQL queries and questions for counting rows with NULL values in specified columns.

        > SELECT COUNT(*) FROM "table" WHERE "col" IS NULL
        > Count the rows where the values of "col" are missing in table "table"

        :param str table_name: The name of the table in the database.
        :param list[str] null_cols: List of column names with NULL values.
        """
        queries = [f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NULL'
                   for col in null_cols]

        questions = [f'Count the rows where the values of "{col}" are missing in table "{table_name}"'
                     for col in null_cols]

        sql_tags = ['NULL-COUNT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def _build_null_not_count(self, table_name, null_cols: list[str]):
        """
        Build SQL queries and questions for counting rows with non-NULL values in specified columns.

        > SELECT COUNT(*) FROM "table" WHERE "col" IS NOT NULL
        > Count the rows where the values of "col" are not missing in table "table"

        :param str table_name: The name of the table in the database.
        :param list[str] null_cols: List of column names with NULL values.
        """
        queries = [f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NOT NULL'
                   for col in null_cols]

        questions = [f'Count the rows where the values of "{col}" are not missing in table "{table_name}"'
                     for col in null_cols]

        sql_tags = ['NOT-NULL-COUNT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def _get_null_cols(self, table_name: str, sample=2):
        """
        Randomly select columns with NULL values from the given table for generating queries.

        :param str table_name: The name of the table in the database.
        :param int sample: Optional. Number of columns to sample. Default is 2.
        :return: List of column names with NULL values.
        """
        df, _, _ = self._sample_cat_num_cols(table_name)
        mask = df.isnull().any()
        cols = list(df.columns[mask])
        return random.sample(cols, sample) if len(cols) >= sample else cols
