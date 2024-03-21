import random

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class NullGenerator(AbstractSqlGenerator):
    """
    A class for generating NULL SQL queries and corresponding questions based on a database table.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generates NULL queries and corresponding questions.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict: A dictionary containing generated SQL tags, queries, and questions.
                Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}

        Examples:
            Given a MultipleDatabases object "database" with a table "table_name" with columns "colors" with 10 NULL values:
            >>> generator = NullGenerator(database)
            >>> generator._build_null_count('table_name', ['colors'])
            >>> generator.sql_generated
            {
                "sql_tags": ['NULL-COUNT'],
                "queries": ['SELECT COUNT(*) FROM "table_name" WHERE "colors" IS NULL'],
                "questions": ['Count the rows where the values of "colors" are missing in table "table_name"']
            }
            >>> generator._build_not_null_count('table_name', ['colors'])
            >>> generator.sql_generated
            {
                "sql_tags": ['NOT-NULL-COUNT'],
                "queries": ['SELECT COUNT(*) FROM "table_name" WHERE "colors" IS NOT NULL'],
                "questions": ['Count the rows where the values of "colors" are not missing in table "table_name"']
            }
        """
        self.empty_sql_generated()
        null_cols = self._get_null_cols(table_name)
        self._build_null_count(table_name, null_cols)
        self._build_not_null_count(table_name, null_cols)
        # self._build_null_with_no_null_col(table_name)
        return self.sql_generated

    def _build_null_count(self, table_name: str, null_cols: list[str]):
        """
        Build SQL queries and questions for counting rows with NULL values in specified columns.

        Args:
            table_name (str): The name of the table in the database.
            null_cols (list): List of column names with NULL values.
        """
        queries = [f'SELECT COUNT(*) FROM `{table_name}` WHERE `{col}` IS NULL'
                   for col in null_cols]

        questions = [f'Count the rows where the values of "{col}" are missing in table "{table_name}"'
                     for col in null_cols]

        sql_tags = ['NULL-COUNT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def _build_not_null_count(self, table_name, null_cols: list[str]):
        """
        Build SQL queries and questions for counting rows with non-NULL values in specified columns.

        Args:
            table_name (str): The name of the table in the database.
            null_cols (list): List of column names with NULL values.
        """
        queries = [f'SELECT COUNT(*) FROM `{table_name}` WHERE `{col}` IS NOT NULL'
                   for col in null_cols]

        questions = [f'Count the rows where the values of "{col}" are not missing in table "{table_name}"'
                     for col in null_cols]

        sql_tags = ['NOT-NULL-COUNT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def _get_null_cols(self, table_name: str, sample=2):
        def _get_sample(_columns, k):
            random.seed(self.seed)
            return random.sample(_columns, k)

        """
        Randomly select columns with NULL values from the given table for generating queries.

        Args:
            table_name (str): The name of the table in the database.
            sample (int, optional): Number of columns to sample. Default is 2.

        Returns:
            list: List of column names with NULL values.
        """
        df, _, _ = self._sample_cat_num_cols(table_name)
        mask = df.isnull().any()
        cols = list(df.columns[mask])
        return _get_sample(cols, sample) if len(cols) >= sample else cols
