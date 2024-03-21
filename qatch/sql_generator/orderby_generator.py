from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class OrderByGenerator(AbstractSqlGenerator):
    """
    A class for generating ORDER BY SQL queries and corresponding questions based on a database table.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generate ORDER BY queries and corresponding questions based on the specified table.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict: A dictionary containing generated SQL tags, queries, and questions.
                Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}

        Examples:
            Given a MultipleDatabases object "database" with a table "table_name" with columns "colors" and "numbers"
            >>> generator = OrderByGenerator(database)
            >>> generator._generate_order_asc("table_name", ["colors", "numbers"])
            >>> generator.sql_generated
            {
                "sql_tags": ["ORDERBY-SINGLE", "ORDERBY-SINGLE"],
                "queries": [
                    'SELECT * FROM "table_name" ORDER BY "colors" ASC',
                    'SELECT * FROM "table_name" ORDER BY "numbers" ASC'
                ],
                "questions": [
                    'Show all data ordered by "colors" in ascending order for the table "table_name"',
                    'Show all data ordered by "numbers" in ascending order for the table "table_name"'
                ]
            }
            >>> generator._generate_order_desc("table_name", ["colors", "numbers"])
            >>> generator.sql_generated
            {
                "sql_tags": ["ORDERBY-SINGLE", "ORDERBY-SINGLE"],
                "queries": [
                    'SELECT * FROM "table_name" ORDER BY "colors" DESC',
                    'SELECT * FROM "table_name" ORDER BY "numbers" DESC'
                ],
                "questions": [
                    'Show all data ordered by "colors" in descending order for the table "table_name"',
                    'Show all data ordered by "numbers" in descending order for the table "table_name"'
                ]
            }
            >>> generator._generate_order_asc_project("table_name", ["colors", "numbers"])
            >>> generator.sql_generated
            {
                "sql_tags": ["ORDERBY-PROJECT", "ORDERBY-PROJECT"],
                "queries": [
                    'SELECT "colors" FROM "table_name" ORDER BY "colors" ASC',
                    'SELECT "numbers" FROM "table_name" ORDER BY "numbers" ASC'
                ],
                "questions": [
                    'Project the "colors" ordered in ascending order for the table "table_name"',
                    'Project the "numbers" ordered in ascending order for the table "table_name"'
                ]
            }
            >>> generator._generate_order_desc_project("table_name", ["colors", "numbers"])
            >>> generator.sql_generated
            {
                "sql_tags": ["ORDERBY-PROJECT", "ORDERBY-PROJECT"],
                "queries": [
                    'SELECT "colors" FROM "table_name" ORDER BY "colors" DESC',
                    'SELECT "numbers" FROM "table_name" ORDER BY "numbers" DESC'
                ],
                "questions": [
                    'Project the "colors" ordered in descending order for the table "table_name"',
                    'Project the "numbers" ordered in descending order for the table "table_name"'
                ]
            }
        """
        self.empty_sql_generated()
        # to avoid too many ORDERBY sql queries, sample only 2 Cate and 2 numerical columns
        _, cat_cols, num_cols = self._sample_cat_num_cols(table_name, sample=2)
        columns = cat_cols + num_cols
        self._generate_order_asc(table_name, columns)
        self._generate_order_desc(table_name, columns)

        self._generate_order_asc_project(table_name, columns)

        self._generate_order_desc_project(table_name, columns)
        return self.sql_generated

    def _generate_order_asc(self, table_name: str, columns: list[str]):
        """
        Generates SQL queries and questions for ordering data in ascending order for each column.

        Args:
            table_name (str): The name of the table in the database.
            columns (list): List of column names.
        """
        queries = [f'SELECT * FROM `{table_name}` ORDER BY `{col}` ASC'
                   for col in columns]

        questions = [
            f'Show all data ordered by "{col}" in ascending order for the table "{table_name}"'
            for col in columns
        ]
        sql_tags = ['ORDERBY-SINGLE'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def _generate_order_desc(self, table_name: str, columns: list[str]):
        """
        Generates SQL queries and questions for ordering data in descending order for each column.

        Args:
            table_name (str): The name of the table in the database.
            columns (list): List of column names.
        """
        queries = [f'SELECT * FROM `{table_name}` ORDER BY `{col}` DESC'
                   for col in columns]

        questions = [
            f'Show all data ordered by {col} in descending order for the table {table_name}'
            for col in columns
        ]
        sql_tags = ['ORDERBY-SINGLE'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def _generate_order_asc_project(self, table_name: str, columns: list[str]):
        """
        Generates SQL queries and questions for projecting a single column and ordering it in ascending order.

        Args:
            table_name (str): The name of the table in the database.
            columns (list): List of column names.
        """
        queries = [f'SELECT `{col}` FROM `{table_name}` ORDER BY `{col}` ASC'
                   for col in columns]

        questions = [
            f'Project the "{col}" ordered in ascending order for the table {table_name}'
            for col in columns
        ]
        sql_tags = ['ORDERBY-PROJECT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)

    def _generate_order_desc_project(self, table_name: str, columns: list[str]):
        """
        Generates SQL queries and questions for projecting a single column and ordering it in descending order.

        Args:
            table_name (str): The name of the table in the database.
            columns (list): List of column names.
        """
        queries = [f'SELECT `{col}` FROM `{table_name}` ORDER BY `{col}` DESC'
                   for col in columns]

        questions = [
            f'Project the "{col}" ordered in descending order for the table {table_name}'
            for col in columns
        ]
        sql_tags = ['ORDERBY-PROJECT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions)
