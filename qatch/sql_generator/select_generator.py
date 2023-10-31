from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class SelectGenerator(AbstractSqlGenerator):
    """
    A class for generating SELECT SQL queries and corresponding questions based on a database table.

    :ivar SingleDatabase database: The SingleDatabase object representing the database to generate queries from.
    :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, and questions.
        Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Abstract method to generate SQL tags, queries, and questions based on the specified table.
        :param str table_name: The name of the table in the database.
        :return: A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}

        """
        self.empty_sql_generated()
        self._select_all_table(table_name)
        self._select_add_col(table_name)
        self._select_random_col(table_name)
        return self.sql_generated

    def _select_all_table(self, table_name: str):
        """Generate the SQL query and question for selecting all rows in the table."""
        sql_tag = ['SELECT-ALL']
        query = [f'SELECT * FROM "{table_name}"']
        question = [f"Show all the rows in the table {table_name}"]
        self.append_sql_generated(sql_tags=sql_tag, queries=query, questions=question)

    def _select_add_col(self, table_name: str):
        """Generate the SQL query and question for selecting increasingly more columns in the table."""
        columns = self.database.get_schema_given(table_name).name.tolist()
        comb_cols_add = self._comb_add_columns(columns)

        questions = self._build_questions(comb_cols_add, table_name)
        queries = self._build_queries(comb_cols_add, table_name)

        self.append_sql_generated(sql_tags=['SELECT-ADD-COL'] * len(comb_cols_add),
                                  queries=queries, questions=questions)

    def _select_random_col(self, table_name: str):
        """Generate the SQL query and question for selecting random columns in the table."""
        columns = self.database.get_schema_given(table_name).name.tolist()
        comb_cols_rand = self._comb_random(columns)

        questions = self._build_questions(comb_cols_rand, table_name)
        queries = self._build_queries(comb_cols_rand, table_name)

        self.append_sql_generated(sql_tags=['SELECT-RANDOM-COL'] * len(comb_cols_rand),
                                  queries=queries, questions=questions)

    def _build_questions(self, combinations: list[list[str]], table_name) -> list[str]:
        """
        Builds questions corresponding to the given column combinations and table name.

        :param list[list[str]] combinations: List of column combinations.
        :param str table_name: The name of the table in the database.

        :return: A list of questions corresponding to the column combinations.
        """
        return [f'Show all {self._get_col_comb_str(comb)} in the table {table_name}'
                for comb in combinations]

    def _build_queries(self, combinations: list[list[str]], table_name: str) -> list[str]:
        """
        Builds SQL queries corresponding to the given column combinations and table name.
        :param list[list[str]] combinations: List of column combinations.
        :param str table_name: The name of the table in the database.
        :return: A list of SQL queries corresponding to the column combinations.
        """
        return [f'SELECT {self._get_col_comb_str(comb)} FROM "{table_name}"'
                for comb in combinations]

    @staticmethod
    def _comb_add_columns(columns: list[str]) -> list[list[str]]:
        """
        Generates column combinations by incrementally adding columns to the query.
        :param list[str] columns: List of column names.
        :return: A list of column combinations.
        """
        return [columns[:i] for i in range(1, len(columns))]
