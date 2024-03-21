from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class SelectGenerator(AbstractSqlGenerator):
    """
    A class for generating SELECT SQL queries and corresponding questions based on a database table.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generate SQL tags, queries, and questions based on the specified table.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict: A dictionary containing generated SQL tags, queries, and questions.
                Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}

        Examples:
            Given a MultipleDatabases object "database" with a table "table_name" with columns "name", "colors" and "numbers":
            >>> generator = SelectGenerator(database)
            >>> generator._select_all_table("table_name")
            >>> generator.sql_generated
            {
                "sql_tags": ["SELECT-ALL"],
                "queries": ['SELECT * FROM "table_name"'],
                "questions": ["Show all the rows in the table table_name"]
            }
            >>> generator._select_add_col("table_name")
            >>> generator.sql_generated
            {
                "sql_tags": ["SELECT-ADD-COL", "SELECT-ADD-COL", "SELECT-ADD-COL"],
                "queries": ['SELECT "colors" FROM "table_name"',
                            'SELECT "colors", "numbers" FROM "table_name"'
                            'SELECT "colors", "numbers", "name" FROM "table_name"'],
                "questions": ["Show all colors in the table table_name",
                              "Show all colors, numbers in the table table_name"
                              "Show all colors, numbers, name in the table table_name"]
            }
            >>> generator._select_random_col("table_name")
            >>> generator.sql_generated
            {
                "sql_tags": ["SELECT-RANDOM-COL", "SELECT-RANDOM-COL"],
                "queries": ['SELECT "colors" FROM "table_name"',
                            'SELECT "name", "numbers" FROM "table_name"',
                            'SELECT "numbers", "colors", "name" FROM "table_name"'],
                "questions": ["Show all colors in the table table_name",
                              "Show all name, numbers in the table table_name",
                              "Show all numbers, colors, name in the table table_name"]
            }

        """
        self.empty_sql_generated()
        self._select_all_table(table_name)
        self._select_add_col(table_name)
        self._select_random_col(table_name)
        return self.sql_generated

    def _select_single_col(self, table_name):
        """Generates SELECT SQL queries and corresponding questions for a single column of a given table.

        This method selects each column of a given table one by one and generates SQL queries
        and corresponding questions. These queries and questions are stored in the 'sql_generated' dictionary attribute of the class.

        Args:
            table_name (str): Name of the table.
        Example:
            Assume the 'users' table has two columns 'id' and 'name'.

                >>> self.database = SingleDatabase('sqlite:///my_db.sqlite3')
                >>> sql_gen = SelectGenerator(self.database)
                >>> sql_gen._select_single_col('users')
                ...
                >>> print(sql_gen.sql_generated)
                {
                    "sql_tags": ["SELECT-SINGLE-COL", "SELECT-SINGLE-COL"],
                    "queries": ["SELECT "id" FROM users;", "SELECT "name" FROM users;"],
                    "questions": ["What are the "id" for all users?", "What are the "name" of all users?"]
                }
        """
        columns = self.database.get_schema_given(table_name).name.tolist()
        columns = [[col] for col in columns]
        # sort columns
        questions = self._build_questions(columns, table_name)
        queries = self._build_queries(columns, table_name)
        self.append_sql_generated(sql_tags=['SELECT-SINGLE-COL'] * len(queries),
                                  queries=queries,
                                  questions=questions)
        return self.sql_generated

    def _select_all_table(self, table_name: str):
        """
        Generate the SQL query and question for selecting all rows in the table.

        Args:
            table_name (str): The name of the table in the database.
        """
        sql_tag = ['SELECT-ALL']
        query = [f'SELECT * FROM `{table_name}`']
        question = [f"Show all the rows in the table {table_name}"]
        self.append_sql_generated(sql_tags=sql_tag, queries=query, questions=question)

    def _select_add_col(self, table_name: str):
        """
        Generate the SQL query and question for selecting increasingly more columns in the table.

        Args:
            table_name (str): The name of the table in the database.
        """
        columns = self.database.get_schema_given(table_name).name.tolist()
        comb_cols_add = self._comb_add_columns(columns)

        questions = self._build_questions(comb_cols_add, table_name)
        queries = self._build_queries(comb_cols_add, table_name)

        self.append_sql_generated(sql_tags=['SELECT-ADD-COL'] * len(comb_cols_add),
                                  queries=queries, questions=questions)

    def _select_random_col(self, table_name: str):
        """
        Generate the SQL query and question for selecting random columns in the table.

        Args:
            table_name (str): The name of the table in the database.
        """
        columns = self.database.get_schema_given(table_name).name.tolist()
        comb_cols_rand = self._comb_random(columns)

        questions = self._build_questions(comb_cols_rand, table_name)
        queries = self._build_queries(comb_cols_rand, table_name)

        self.append_sql_generated(sql_tags=['SELECT-RANDOM-COL'] * len(comb_cols_rand),
                                  queries=queries, questions=questions)

    def _build_questions(self, combinations: list[list[str]], table_name) -> list[str]:
        """
        Builds questions corresponding to the given column combinations and table name.

        Args:
            combinations (list[list[str]]): List of column combinations.
            table_name (str): The name of the table in the database.

        Returns:
            list[str]: A list of questions corresponding to the column combinations.
        """
        return [f'Show all {self._get_col_comb_str(comb)} in the table {table_name}'
                for comb in combinations]

    def _build_queries(self, combinations: list[list[str]], table_name: str) -> list[str]:
        """
        Builds SQL queries corresponding to the given column combinations and table name.

        Args:
            combinations (list[list[str]]): List of column combinations.
            table_name (str): The name of the table in the database.

        Returns:
            list[str]: A list of SQL queries corresponding to the column combinations.
        """
        return [f'SELECT {self._get_col_comb_str(comb)} FROM `{table_name}`'
                for comb in combinations]

    @staticmethod
    def _comb_add_columns(columns: list[str]) -> list[list[str]]:
        """
        Generates column combinations by incrementally adding columns to the query.

        Args:
            columns (list[str]): List of column names.

        Returns:
            list[list[str]]: A list of column combinations.
        """
        return [columns[:i] for i in range(1, len(columns))]
