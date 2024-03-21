from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class GroupByGenerator(AbstractSqlGenerator):
    """
    A class for generating SQL queries and corresponding questions based on group-by operations
    performed on categorical and numerical columns of a database table.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generates Group By queries and corresponding questions for both categorical and numerical columns.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict: A dictionary containing generated SQL tags, queries, and questions.
                Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}

        Examples:
            Given a MultipleDatabases object "database" with a table "table_name" with columns "colors" and "numbers"
            >>> generator = GroupByGenerator(database)
            >>> generator._build_group_by_no_agg("table_name", ["colors"])
            >>> generator.sql_generated
            {
                "sql_tags": ["GROUPBY-NO-AGGR"],
                "queries": ["SELECT `colors` FROM `table_name` GROUP BY `colors`"],
                "questions": ["Show all \"colors\" in the table "table_name" for each \"colors\""]
            }
            >>> generator._build_group_by_with_count("table_name", ["colors"])
            >>> generator.sql_generated
            {
                "sql_tags": ["GROUPBY-COUNT"],
                "queries": ["SELECT `colors`, COUNT(*) FROM `table_name` GROUP BY `colors`"],
                "questions": ["For each \"colors\", count the number of rows in table "table_name""]
            }
            >>> generator._build_group_by_with_agg("table_name")
            >>> generator.sql_generated
            {
                "sql_tags": ["GROUPBY-AGG-MIN", "GROUPBY-AGG-MAX", "GROUPBY-AGG-AVG", "GROUPBY-AGG-SUM"],
                "queries": [
                    "SELECT `colors`, MIN(`numbers`) FROM `table_name` GROUP BY `colors`",
                    "SELECT `colors`, MAX(`numbers`) FROM `table_name` GROUP BY `colors`",
                    "SELECT `colors`, AVG(`numbers`) FROM `table_name` GROUP BY `colors`",
                    "SELECT `colors`, SUM(`numbers`) FROM `table_name` GROUP BY `colors`"
                ],
                "questions": [
                    "For each `colors`, find the min of `numbers` in table `table_name`",
                    "For each `colors`, find the max of `numbers` in table `table_name`",
                    "For each `colors`, find the avg of `numbers` in table `table_name`",
                    "For each `colors`, find the sum of `numbers` in table `table_name`"
                ]
            }
        """
        self.empty_sql_generated()
        df, cat_cols, num_cols = self._sample_cat_num_cols(table_name)
        self._build_group_by_no_agg(table_name, cat_cols)
        self._build_group_by_with_count(table_name, cat_cols)
        self._build_group_by_with_agg(table_name)
        return self.sql_generated

    def _build_group_by_no_agg(self, table_name: str, cat_cols: list):
        """
        Generate group-by SQL queries and questions without aggregation
        for random combinations of categorical columns.
        The query result is the same as Distinct.

        Args:
            table_name (str): The name of the table in the database.
            cat_cols (List[str]): List of categorical columns.
        """
        random_combinations = self._comb_random(cat_cols)

        questions = [f'Show all {self._get_col_comb_str(comb)}' \
                     f' in the table "{table_name}" for each {self._get_col_comb_str(comb)}'
                     for comb in random_combinations]

        queries = [f'SELECT {self._get_col_comb_str(comb)} FROM ' \
                   f'`{table_name}` GROUP BY {self._get_col_comb_str(comb)}'
                   for comb in random_combinations]

        sql_tags = ['GROUPBY-NO-AGGR'] * len(queries)

        self.append_sql_generated(sql_tags=sql_tags, queries=queries,
                                  questions=questions)

    def _build_group_by_with_count(self, table_name: str, cat_cols: list):
        """
        Generate group-by SQL queries and questions with count aggregation for categorical columns.

        Args:
            table_name (str): The name of the table in the database.
            cat_cols (List[str]): List of categorical columns.
        """
        questions = [f'For each "{col}", count the number of rows in table "{table_name}"'
                     for col in cat_cols]
        queries = [f'SELECT `{col}`, COUNT(*) FROM `{table_name}` GROUP BY `{col}`'
                   for col in cat_cols]
        sql_tags = ['GROUPBY-COUNT'] * len(queries)

        self.append_sql_generated(sql_tags=sql_tags, queries=queries,
                                  questions=questions)

    def _build_group_by_with_agg(self, table_name: str):
        """
        Generate group-by SQL queries and questions with aggregation for numerical columns.

        Args:
            table_name (str): The name of the table in the database.
        """
        # with sample == 2 we get 4 tests for each aggregation -> 4*4 = 16 tests
        # with sample == 3 we get 9 tests for each aggregation -> 9*4 = 36 tests
        _, cat_cols, num_cols = self._sample_cat_num_cols(table_name, sample=2)
        for agg in ['min', 'max', 'avg', 'sum']:
            questions = [f'For each "{c_col}", find the {agg} of "{n_col}" in table "{table_name}"'
                         for c_col in cat_cols
                         for n_col in num_cols]

            queries = [f'SELECT `{c_col}`, {agg.upper()}(`{n_col}`) FROM `{table_name}` GROUP BY `{c_col}`'
                       for c_col in cat_cols
                       for n_col in num_cols]

            sql_tags = [f'GROUPBY-AGG-{agg.upper()}'] * len(queries)

            self.append_sql_generated(sql_tags=sql_tags, queries=queries,
                                      questions=questions)
