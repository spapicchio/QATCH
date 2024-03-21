from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class SimpleAggGenerator(AbstractSqlGenerator):
    """
    A class for generating Simple Aggregation queries and corresponding questions based on a database table.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": List[str], "queries": List[str], "questions": List[str]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generates Simple Aggregation SQL queries and corresponding questions for the specified table.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict: A dictionary containing generated SQL tags, queries, and questions.
                  Format: {"sql_tags": List[str], "queries": List[str], "questions": List[str]}

        Examples:
            Given a MultipleDatabases object "database" with a table "table_name" with columns "colors" and "numbers"
            >>> generator = SimpleAggGenerator(database)
            >>> generator._build_count_cat("table_name", ["colors"])
            >>> generator.sql_generated
            {
                "sql_tags": ["SIMPLE-AGG-COUNT", "SIMPLE-AGG-COUNT-DISTINCT"],
                "queries": [
                    'SELECT COUNT(*) FROM "table_name"',
                    'SELECT COUNT(DISTINCT"colors") FROM "table_name"'
                ],
                "questions": [
                    'Count the records in table "table_name"?',
                    'How many different "colors" are in table "table_name"?'
                ]
            }
            >>> generator._build_count_agg("table_name", ["numbers"])
            >>> generator.sql_generated
            {
                "sql_tags": ["SIMPLE-AGG-MAX", "SIMPLE-AGG-MIN", "SIMPLE-AGG-AVG"],
                "queries": [
                    'SELECT MAX("numbers") FROM "table_name"',
                    'SELECT MIN("numbers") FROM "table_name"',
                    'SELECT AVG("numbers") FROM "table_name"'
                ],
                "questions": [
                    'Find the maximum "numbers" for the table "table_name"',
                    'Find the minimum "numbers" for the table "table_name"',
                    'Find the average "numbers" for the table "table_name"'
                ]
            }
        """
        self.empty_sql_generated()
        _, cat_cols, num_cols = self._sample_cat_num_cols(table_name)
        self._build_count_cat(table_name, cat_cols)
        self._build_count_agg(table_name, num_cols)
        return self.sql_generated

    def _build_count_cat(self, table_name, cat_cols):
        """
        Generates COUNT SQL queries and questions for categorical columns.

        Args:
            table_name (str): The name of the table in the database.
            cat_cols (List[str]): List of categorical columns in the table.
        """

        queries = [f'SELECT COUNT(*) FROM `{table_name}`']
        questions = [f'Count the records in table "{table_name}"?']
        sql_tags = ['SIMPLE-AGG-COUNT']

        for cat_col in cat_cols:
            queries += [f'SELECT COUNT(DISTINCT `{cat_col}`) FROM `{table_name}`']
            questions += [f'How many different "{cat_col}" are in table "{table_name}"?']
            sql_tags += ['SIMPLE-AGG-COUNT-DISTINCT']

        self.append_sql_generated(sql_tags, queries, questions)

    def _build_count_agg(self, table_name, num_cols):
        """
        Generates MAX, MIN, and AVG SQL queries and questions for numerical columns.

        Args:
            table_name (str): The name of the table in the database.
            num_cols (List[str]): List of numerical columns in the table.
        """
        for num_col in num_cols:
            queries = [
                f'SELECT MAX(`{num_col}`) FROM `{table_name}`',
                f'SELECT MIN(`{num_col}`) FROM `{table_name}`',
                f'SELECT AVG(`{num_col}`) FROM `{table_name}`'
            ]
            questions = [
                f'Find the maximum "{num_col}" for the table "{table_name}"',
                f'Find the minimum "{num_col}" for the table "{table_name}"',
                f'Find the average "{num_col}" for the table "{table_name}"'
            ]
            sql_tags = ['SIMPLE-AGG-MAX', 'SIMPLE-AGG-MIN', 'SIMPLE-AGG-AVG']
            self.append_sql_generated(sql_tags, queries, questions)
