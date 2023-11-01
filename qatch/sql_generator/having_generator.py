import pandas as pd

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class HavingGenerator(AbstractSqlGenerator):
    """
    A class for generating HAVING SQL queries and corresponding questions based on a database table.

    :ivar SingleDatabase database: The SingleDatabase object representing the database to generate queries from.
    :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, and questions.
        Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def __init__(self, database: SingleDatabase, seed=2023):
        super().__init__(database, seed)

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generate HAVING SQL queries and corresponding questions for categorical and numerical columns.

        :param str table_name: The name of the table in the database.
        :return: A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
        """
        self.empty_sql_generated()
        df, cat_cols, num_cols = self._sample_cat_num_cols(table_name)
        self._build_having_count(table_name, cat_cols, df)
        self._build_having_agg(table_name, cat_cols, num_cols, df)
        return self.sql_generated

    def _build_having_count(self, table_name: str, cat_cols: list, df: pd.DataFrame):
        """
        Build HAVING SQL queries and questions for categorical columns based on row counts.

        > SELECT policy_type_code FROM policies GROUP BY policy_type_code HAVING count(*)>2
        > Find all the policy types that have more than 2 records

        :param str table_name: The name of the table in the database.
        :param list cat_cols: List of categorical columns.
        """

        for cat_col in cat_cols:
            # get a mean count of the category cat_col
            mean_count = self._get_average_of_count_cat_col(table_name, cat_col)
            # int(df.groupby(cat_col).count().mean().values[0])
            queries = [
                f"""SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) >= '{mean_count}'""",
                f"""SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) <= '{mean_count}'""",
                f"""SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) = '{mean_count}'"""
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

        > SELECT Product_Name FROM PRODUCTS GROUP BY Product_Name HAVING avg(Product_Price) < 1000000
        > Find the product names whose average product price is below 1000000.

        :param str table_name: The name of the table in the database.
        :param list cat_cols: List of categorical columns.
        :param list num_cols: List of numerical columns.
        :param pd.DataFrame df: The DataFrame containing the data.
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
                    f"""SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING AVG("{num_col}") >= '{mean_mean_mean}'""",
                    f"""SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING AVG("{num_col}") <= '{mean_mean_mean}'""",
                    f"""SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING SUM("{num_col}") >= '{mean_mean_sum}'""",
                    f"""SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING SUM("{num_col}") <= '{mean_mean_sum}'""",
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
        # TODO: pandas performs faster when number of tuples is 5e4 or more
        # SQL query to get the average count for each category
        inner_query = f'SELECT COUNT(*) AS row_count FROM "{table_name}" GROUP BY "{cat_col}"'
        # Run the inner query and get the average of row counts
        average = self.database.run_query(f'SELECT AVG(row_count) FROM ({inner_query})')[0][0]
        return int(average)

    def _get_average_of_sum_avg_cat_col(self, table_name, cat_col, num_col):
        # TODO: pandas performs faster when number of tuples is 5e4 or more
        # SQL queries to get the average sum and average of numerical column for each category
        inner_query_sum = f'SELECT SUM("{num_col}") AS sum_col FROM "{table_name}" GROUP BY "{cat_col}"'
        inner_query_avg = f'SELECT AVG("{num_col}") AS avg_col FROM "{table_name}" GROUP BY "{cat_col}"'
        # Run the inner queries and get the average of sums and averages
        average_sum = self.database.run_query(f'SELECT AVG(sum_col) FROM ({inner_query_sum})')[0][0]
        average_avg = self.database.run_query(f'SELECT AVG(avg_col) FROM ({inner_query_avg})')[0][0]
        return round(average_sum, 2), round(average_avg, 2)
