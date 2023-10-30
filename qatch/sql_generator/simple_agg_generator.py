from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class SimpleAggGenerator(AbstractSqlGenerator):
    """
    A class for generating Simple Aggregation queries and corresponding questions based on a database table.

    :ivar SingleDatabase database: The SingleDatabase object representing the database to generate queries from.
    :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, and questions.
        Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def __init__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
         Abstract method to generate SQL tags, queries, and questions based on the specified table.
         :param str table_name: The name of the table in the database.
         :return: A dictionary containing generated SQL tags, queries, and questions.
             Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
         """
        self.empty_sql_generated()
        _, cat_cols, num_cols = self._sample_cat_num_cols(table_name)
        self._build_count_cat(table_name, cat_cols)
        self._build_count_agg(table_name, num_cols)
        return self.sql_generated

    def _build_count_cat(self, table_name, cat_cols):
        """
        Generates COUNT SQL queries and questions for categorical columns.
        > SELECT COUNT(*) FROM table_name
        > Count the records in Customers_Cards?

        > SELECT count(DISTINCT card_type_code) FROM Customers_Cards
        > How many different card types are there?

        :param str table_name: The name of the table in the database.
        :param list cat_cols: List of categorical columns in the table.
        """

        queries = [f'SELECT COUNT(*) FROM "{table_name}"']
        questions = [f'Count the records in table "{table_name}"?']
        sql_tags = ['SIMPLE-AGG-COUNT']

        for cat_col in cat_cols:
            queries += [f'SELECT COUNT(DISTINCT"{cat_col}") FROM "{table_name}"']
            questions += [f'How many different "{cat_col}" are in table "{table_name}"?']
            sql_tags += ['SIMPLE-AGG-COUNT-DISTINCT']

        self.append_sql_generated(sql_tags, queries, questions)

    def _build_count_agg(self, table_name, num_cols):
        """
        Generates MAX, MIN, and AVG SQL queries and questions for numerical columns.

        SELECT max(monthly_rental) FROM Student_Addresses
        Find the maximum monthly rental for the table Student_Addresses.

        :param str table_name: The name of the table in the database.
        :param list num_cols: List of numerical columns in the table.
        """
        for num_col in num_cols:
            queries = [
                f'SELECT MAX("{num_col}") FROM "{table_name}"',
                f'SELECT MIN("{num_col}") FROM "{table_name}"',
                f'SELECT AVG("{num_col}") FROM "{table_name}"'
            ]
            questions = [
                f'Find the maximum "{num_col}" for the table "{table_name}"',
                f'Find the minimum "{num_col}" for the table "{table_name}"',
                f'Find the average "{num_col}" for the table "{table_name}"'
            ]
            sql_tags = ['SIMPLE-AGG-MAX', 'SIMPLE-AGG-MIN', 'SIMPLE-AGG-AVG']
            self.append_sql_generated(sql_tags, queries, questions)
