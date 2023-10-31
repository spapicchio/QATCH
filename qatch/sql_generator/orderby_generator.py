from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class OrderByGenerator(AbstractSqlGenerator):
    """
    A class for generating ORDER BY SQL queries and corresponding questions based on a database table.

    :ivar SingleDatabase database: The SingleDatabase object representing the database to generate queries from.
    :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, and questions.
        Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)
        self.empty_sql_generated()

    def sql_generate(self, table_name: str):
        """
        Abstract method to generate SQL tags, queries, and questions based on the specified table.
        :param str table_name: The name of the table in the database.
        :return: A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
        """
        self.empty_sql_generated()
        columns = self.database.get_schema_given(table_name).name.tolist()
        self.generate_order_asc(table_name, columns)
        self.generate_order_desc(table_name, columns)
        # TODO check when necessary
        # self.generate_order_asc_project(table_name, columns)
        # self.generate_order_desc_project(table_name, columns)
        return self.sql_generated

    def generate_order_asc(self, table_name: str, columns: list[str]):
        """
        Generates SQL queries and questions for ordering data in ascending order for each column.

        :param str table_name: The name of the table in the database.
        :param list[str] columns: List of column names.
        """
        queries = [f'SELECT * FROM "{table_name}" ORDER BY "{col}" ASC'
                   for col in columns]

        questions = [
            f'Show all data ordered by "{col}" in ascending order for the table "{table_name}"'
            for col in columns
        ]
        sql_tags = ['ORDERBY-SINGLE'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def generate_order_desc(self, table_name, columns: list[str]):
        """
        Generates SQL queries and questions for ordering data in descending order for each column.

        :param str table_name: The name of the table in the database.
        :param list[str] columns: List of column names.
        """
        queries = [f'SELECT * FROM "{table_name}" ORDER BY "{col}" DESC'
                   for col in columns]

        questions = [
            f'Show all data ordered by "{col}" in descending order for the table "{table_name}"'
            for col in columns
        ]
        sql_tags = ['ORDERBY-SINGLE'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def generate_order_asc_project(self, table_name: str, columns: list[str]):
        """
        Generates SQL queries and questions for projecting a single column and ordering it in ascending order.

        :param str table_name: The name of the table in the database.
        :param list[str] columns: List of column names.
        """
        queries = [f'SELECT "{col}" FROM "{table_name}" ORDER BY "{col}" ASC'
                   for col in columns]

        questions = [
            f'Project the "{col}" ordered in ascending order for the table {table_name}'
            for col in columns
        ]
        sql_tags = ['ORDERBY-PROJECT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def generate_order_desc_project(self, table_name, columns: list[str]):
        """
        Generates SQL queries and questions for projecting a single column and ordering it in descending order.

        :param str table_name: The name of the table in the database.
        :param list[str] columns: List of column names.
        """
        queries = [f'SELECT "{col}" FROM "{table_name}" ORDER BY "{col}" DESC'
                   for col in columns]

        questions = [
            f'Project the "{col}" ordered in descending order for the table {table_name}'
            for col in columns
        ]
        sql_tags = ['ORDERBY-PROJECT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)
