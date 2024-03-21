from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class DistinctGenerator(AbstractSqlGenerator):
    """
    A class for generating DISTINCT SQL queries and corresponding questions based on
    categorical columns of a database table.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generates DISTINCT SQL queries and corresponding questions based on categorical columns of a table.
        Generates two distinct tags: DISTINCT-SINGLE and DISTINCT-MULT only for the categorical columns.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict: A dictionary containing generated SQL tags, queries, and questions.
                Format: {"sql_tags": List[str], "queries": List[str], "questions": List[str]}

        Examples:
            Given a MultipleDatabases object "database" with a table "table_name" with columns "colors" and "names"
            >>> generator = DistinctGenerator(database)
            >>> generator._distinct_single_col("table_name", ["colors"])
            >>> generator.sql_generated
            {"sql_tags": ["DISTINCT-SINGLE"],
            "queries": ["SELECT DISTINCT \"colors\" FROM \"table_name\""],
            "questions": ["Show the different \"colors\" in the table table_name"]}
            >>> generator_distinct_mult_col("table_name", ["colors", "names"])
            >>> generator.sql_generated
            {"sql_tags": ["DISTINCT-MULT"],
            "queries": ["SELECT DISTINCT \"colors\", \"names\" FROM \"table_name\""],
            "questions": ["Show the different \"colors\", \"names\" in the table table_name"]}
        """
        self.empty_sql_generated()
        _, cat_cols, _ = self._sample_cat_num_cols(table_name)
        self._distinct_single_col(table_name, cat_cols)
        self._distinct_mult_col(table_name, cat_cols)
        return self.sql_generated

    def _distinct_single_col(self, table_name: str, cat_columns: list):
        """
        Generates DISTINCT SQL queries and questions for individual categorical columns.

        Args:
            table_name (str): The name of the table in the database.
            cat_columns (List[str]): List of categorical column names.
        """
        queries = [f'SELECT DISTINCT `{col}` FROM `{table_name}`'
                   for col in cat_columns]

        questions = [f'Show the different "{col}" in the table "{table_name}"'
                     for col in cat_columns]

        sql_tags = ['DISTINCT-SINGLE'] * len(queries)
        self.append_sql_generated(sql_tags=sql_tags, queries=queries, questions=questions)

    def _distinct_mult_col(self, table_name: str, cat_columns: list):
        """
        Generates DISTINCT SQL queries and questions for combinations of multiple categorical columns.

        Args:
            table_name (str): The name of the table in the database.
            cat_columns (List[str]): List of categorical column names.
        """
        combinations = self._comb_random(cat_columns)
        queries = [f'SELECT DISTINCT {self._get_col_comb_str(comb)} FROM `{table_name}`'
                   for comb in combinations]

        questions = [f'Show the different {self._get_col_comb_str(comb)} in the table "{table_name}"'
                     for comb in combinations]

        sql_tags = ['DISTINCT-MULT'] * len(queries)
        self.append_sql_generated(sql_tags=sql_tags, queries=queries, questions=questions)
