import pandas as pd

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class HavingGenerator(AbstractSqlGenerator):
    """
    A class for generating HAVING SQL queries and corresponding questions based on a database table.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generate HAVING SQL queries and corresponding questions for categorical and numerical columns.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict: A dictionary containing generated SQL tags, queries, and questions.
                Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}

        Examples:
            Given a MultipleDatabases object "database" with a table "table_name" with columns "colors" and "numbers"
            and (i) the average number of rows for each category in "colors" is 2
            (ii) the mean of the average numbers for each category is 5
            (iii) the average sum of numbers for each category is 10:
            >>> generator = HavingGenerator(database)
            >>> generator._build_having_count(table_name, ["colors"], df)
            >>> generator.sql_generated
            {
                "sql_tags": ["HAVING-COUNT-GR", "HAVING-COUNT-LS", "HAVING-COUNT-EQ"],
                "queries": [
                    'SELECT `colors` FROM `table_name` GROUP BY `colors` HAVING count(*) >= 2',
                    'SELECT `colors` FROM `table_name` GROUP BY `colors` HAVING count(*) <= 2',
                    'SELECT `colors` FROM `table_name` GROUP BY `colors` HAVING count(*) = 2'
                ],
                "questions": [
                    'Find all the `colors` that have at least 2 records in table `table_name`',
                    'Find all the `colors` that have at most 2 records in table `table_name`',
                    'Find all the `colors` that have exactly 2 records in table `table_name`'
                ]
            }
            >>> generator._build_having_agg(table_name, ["colors"], ["numbers"], df)
            >>> generator.sql_generated
            {
                "sql_tags": ["HAVING-AGG-AVG-GR", "HAVING-AGG-AVG-LS", "HAVING-AGG-SUM-GR", "HAVING-AGG-SUM-LS"],
                "queries": [
                    'SELECT `colors` FROM `table_name` GROUP BY `colors` HAVING AVG(`numbers`) >= 5.0',
                    'SELECT `colors` FROM `table_name` GROUP BY `colors` HAVING AVG(`numbers`) <= 5.0',
                    'SELECT `colors` FROM `table_name` GROUP BY `colors` HAVING SUM(`numbers`) >= 10.0',
                    'SELECT `colors` FROM `table_name` GROUP BY `colors` HAVING SUM(`numbers`) <= 10.0'
                ],
                "questions": [
                    'List the `colors` which average `numbers` is at least 5.0 in table `table_name`',
                    'List the `colors` which average `numbers` is at most 5.0 in table `table_name`',
                    'List the `colors` which summation of `numbers` is at least 5.0 in table `table_name`',
                    'List the `colors` which summation of `numbers` is at most 5.0 in table `table_name`'
                ]
            }
        """
        self.empty_sql_generated()
        df, cat_cols, num_cols = self._sample_cat_num_cols(table_name)
        self._build_having_count(table_name, cat_cols, df)
        self._build_having_agg(table_name, cat_cols, num_cols, df)
        return self.sql_generated

    def _build_having_count(self, table_name: str, cat_cols: list, df: pd.DataFrame):
        """
        Build HAVING SQL queries and questions for categorical columns based on row counts.

        Args:
            table_name (str): The name of the table in the database.
            cat_cols (list): List of categorical columns.
            df (pd.DataFrame): The DataFrame containing the data.
        """

        for cat_col in cat_cols:
            # get a mean count of the category cat_col
            mean_count = self._get_average_of_count_cat_col(table_name, cat_col)
            # int(df.groupby(cat_col).count().mean().values[0])
            queries = [
                f"""SELECT `{cat_col}` FROM `{table_name}` GROUP BY `{cat_col}` HAVING count(*) >= {mean_count}""",
                f"""SELECT `{cat_col}` FROM `{table_name}` GROUP BY `{cat_col}` HAVING count(*) <= {mean_count}""",
                f"""SELECT `{cat_col}` FROM `{table_name}` GROUP BY `{cat_col}` HAVING count(*) = {mean_count}"""
            ]

            questions = [
                f'Find all the "{cat_col}" that have at least {mean_count} records in table "{table_name}"',
                f'Find all the "{cat_col}" that have at most {mean_count} records in table "{table_name}"',
                f'Find all the "{cat_col}" that have exactly {mean_count} records in table "{table_name}"'
            ]

            sql_tags = ['HAVING-COUNT-GR', 'HAVING-COUNT-LS', 'HAVING-COUNT-EQ']

            self.append_sql_generated(sql_tags, queries, questions)

    def _build_having_agg(self, table_name: str, cat_cols: list, num_cols: list, df: pd.DataFrame):
        """
        Build HAVING SQL queries and questions for numerical columns based on aggregations.

        Args:
            table_name (str): The name of the table in the database.
            cat_cols (list): List of categorical columns.
            num_cols (list): List of numerical columns.
            df (pd.DataFrame): The DataFrame containing the data.
        """
        # with sample == 2 we get 4 tests for each aggregation -> 4*4 = 16 tests
        # with sample == 3 we get 9 tests for each aggregation -> 9*4 = 36 tests
        for cat_col in cat_cols:
            # the mean for each grouped category
            # mean_sum = df.groupby(cat_col).sum(numeric_only=True)
            # mean_mean = df.groupby(cat_col).mean(numeric_only=True)
            for num_col in num_cols:
                # the mean of sum for the grouped category
                # mean_mean_sum = round(mean_sum[num_col].mean(), 2)
                # mean_mean_mean = round(mean_mean[num_col].mean(), 2)
                mean_mean_sum, mean_mean_mean = self._get_average_of_sum_avg_cat_col(table_name, cat_col, num_col)
                queries = [
                    f"""SELECT `{cat_col}` FROM `{table_name}` GROUP BY `{cat_col}` HAVING AVG(`{num_col}`) >= {mean_mean_mean}""",
                    f"""SELECT `{cat_col}` FROM `{table_name}` GROUP BY `{cat_col}` HAVING AVG(`{num_col}`) <= {mean_mean_mean}""",
                    f"""SELECT `{cat_col}` FROM `{table_name}` GROUP BY `{cat_col}` HAVING SUM(`{num_col}`) >= {mean_mean_sum}""",
                    f"""SELECT `{cat_col}` FROM `{table_name}` GROUP BY `{cat_col}` HAVING SUM(`{num_col}`) <= {mean_mean_sum}""",
                ]

                questions = [
                    f'List the "{cat_col}" which average "{num_col}" is at least {mean_mean_mean} in table "{table_name}"',
                    f'List the "{cat_col}" which average "{num_col}" is at most {mean_mean_mean} in table "{table_name}"',

                    f'List the "{cat_col}" which summation of "{num_col}" is at least {mean_mean_sum} in table "{table_name}"',
                    f'List the "{cat_col}" which summation of "{num_col}" is at most {mean_mean_sum} in table "{table_name}"',
                ]

                sql_tags = ['HAVING-AGG-AVG-GR', 'HAVING-AGG-AVG-LS',
                            'HAVING-AGG-SUM-GR', 'HAVING-AGG-SUM-LS']

                self.append_sql_generated(sql_tags, queries, questions)

    def _get_average_of_count_cat_col(self, table_name, cat_col):
        """
        Helper method to calculate the average count of rows for each category in a categorical column.

        Args:
            table_name (str): The name of the table in the database.
            cat_col (str): The name of the categorical column.

        Returns:
            int: The average count of rows for each category.
        """
        # TODO: pandas performs faster when number of tuples is 5e4 or more
        # SQL query to get the average count for each category
        inner_query = f'SELECT COUNT(*) AS row_count FROM `{table_name}` GROUP BY `{cat_col}`'
        # Run the inner query and get the average of row counts
        average = self.database.run_query(f'SELECT AVG(row_count) FROM ({inner_query})')[0][0]
        return int(average)

    def _get_average_of_sum_avg_cat_col(self, table_name, cat_col, num_col):
        """
        Helper method to calculate the average sum and average of a numerical column for each category in a categorical column.

        Args:
            table_name (str): The name of the table in the database.
            cat_col (str): The name of the categorical column.
            num_col (str): The name of the numerical column.

        Returns:
            tuple: A tuple containing the average sum and average of the numerical column for each category.
        """
        # TODO: pandas performs faster when number of tuples is 5e4 or more
        # SQL queries to get the average sum and average of numerical column for each category
        inner_query_sum = f'SELECT SUM(`{num_col}`) AS sum_col FROM `{table_name}` GROUP BY `{cat_col}`'
        inner_query_avg = f'SELECT AVG(`{num_col}`) AS avg_col FROM `{table_name}` GROUP BY `{cat_col}`'
        # Run the inner queries and get the average of sums and averages
        average_sum = self.database.run_query(f'SELECT AVG(sum_col) FROM ({inner_query_sum})')[0][0]
        average_avg = self.database.run_query(f'SELECT AVG(avg_col) FROM ({inner_query_avg})')[0][0]
        return round(average_sum, 2), round(average_avg, 2)
