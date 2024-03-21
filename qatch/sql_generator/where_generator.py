import numpy as np
import pandas as pd

from .abstract_sql_generator import AbstractSqlGenerator


class WhereGenerator(AbstractSqlGenerator):
    """
    A class for generating WHERE SQL queries and corresponding questions based on a database table.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, questions, and results.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str], "results": list[pd.DataFrame]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generates WHERE SQL queries and corresponding questions for both categorical and numerical columns.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict: A dictionary containing generated SQL tags, queries, questions, and results.
                  Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}

        Examples:
            Given a MultipleDatabases object "database" with a table "table_name" with columns "colors" and "numbers"
            >>> sample_df = pd.DataFrame({"colors": ["green", "blue", "blue", "blue", "blue"], "numbers": [1, 2, 3, 4, 5]})
            >>> generator = WhereGenerator(database)
            >>> generator._generate_where_categorical("table_name", ["colors"], sample_df)
            >>> generator.sql_generated
            {
                "sql_tags": ["WHERE-CAT-MOST-FREQUENT", "WHERE-CAT-LEAST-FREQUENT",
                            'WHERE-CAT-MOST-FREQUENT', 'WHERE-CAT-LEAST-FREQUENT',
                            "WHERE-NOT-MOST-FREQUENT", "WHERE-NOT-LEAST-FREQUENT"],
                "queries": [
                    'SELECT * FROM "table_name" WHERE "colors" == "blue"',
                    'SELECT * FROM "table_name" WHERE "colors" == "green"',
                    'SELECT * FROM "table_name" WHERE "colors" != "blue"',
                    'SELECT * FROM "table_name" WHERE "colors" != "green"',
                    'SELECT * FROM "table_name" WHERE NOT "colors" == "blue"',
                    'SELECT * FROM "table_name" WHERE NOT "colors" == "green"',
                ],
                "questions": [
                    'Show the data of the table "table_name" where "colors" is equal to blue',
                    'Show the data of the table "table_name" where "colors" is equal to green',
                    'Show the data of the table "table_name" where "colors" is different from blue',
                    'Show the data of the table "table_name" where "colors" is different from green',
                    'Show the data of the table "table_name" where "colors" is not equal to blue',
                    'Show the data of the table "table_name" where "colors" is not equal to green',
                ]
            }
            >>> generator._generate_where_numerical("table_name", ["numbers"], sample_df)
            >>> generator.sql_generated
            {
                "sql_tags": ['WHERE-NUM-MAX-VALUES', 'WHERE-NUM-MIN-VALUES',
                            'WHERE-NUM-MEAN-VALUES', 'WHERE-NUM-MEAN-VALUES'],
                "queries": ['SELECT * FROM "table_name" WHERE "numbers" < 5',
                            'SELECT * FROM "table_name" WHERE "numbers" > 1,
                            'SELECT * FROM "table_name" WHERE "numbers" > 3.0'
                            'SELECT * FROM "table_name" WHERE "numbers" < 3.0'],
                "question": ['Show the data of the table "table_name" where "numbers" is less than 5',
                            'Show the data of the table "table_name" where "numbers" is greater than 1',
                            'Show the data of the table "table_name" where "numbers" is greater than 3.0',
                            'Show the data of the table "table_name" where "numbers" is less than 3.0'],
            }
        """
        self.empty_sql_generated()
        df, cat_cols, num_cols = self._sample_cat_num_cols(table_name)
        self._generate_where_categorical(table_name, cat_cols, df)
        self._generate_where_numerical(table_name, num_cols, df)
        return self.sql_generated

    def _generate_where_categorical(self, table_name: str, cat_cols: list, df: pd.DataFrame):
        """
        Generates WHERE SQL queries and questions for categorical columns.
        Generates test for both most frequent and least frequent values.

        Args:
            table_name (str): The name of the table in the database.
            cat_cols (list): List of categorical columns.
            df (pd.DataFrame): The DataFrame containing the data.
        """
        if len(cat_cols) == 0:
            # no categorical attributes present
            return
        most_frequent_elements = [self._get_most_frequent_or_max_value(df[col].values) for col in cat_cols]
        least_frequent_elements = [self._get_least_frequent_or_min_value(df[col].values) for col in cat_cols]
        for col, most_freq, least_freq in zip(cat_cols, most_frequent_elements, least_frequent_elements):
            queries = [
                f"""SELECT * FROM `{table_name}` WHERE `{col}` == `{most_freq}` """,
                f"""SELECT * FROM `{table_name}` WHERE `{col}` == `{least_freq}` """,
                f"""SELECT * FROM `{table_name}` WHERE `{col}` != `{most_freq}` """,
                f"""SELECT * FROM `{table_name}` WHERE `{col}` != `{least_freq}` """,
                f"""SELECT * FROM `{table_name}` WHERE NOT `{col}` == `{most_freq}` """,
                f"""SELECT * FROM `{table_name}` WHERE NOT `{col}` == `{least_freq}` """,
            ]

            questions = [
                f'Show the data of the table "{table_name}" where "{col}" is equal to {most_freq}',
                f'Show the data of the table "{table_name}" where "{col}" is equal to {least_freq}',
                f'Show the data of the table "{table_name}" where "{col}" is different from {most_freq}',
                f'Show the data of the table "{table_name}" where "{col}" is different from {least_freq}',
                f'Show the data of the table "{table_name}" where "{col}" is not equal to {most_freq}',
                f'Show the data of the table "{table_name}" where "{col}" is not equal to {least_freq}',
            ]

            sql_tags = ['WHERE-CAT-MOST-FREQUENT', 'WHERE-CAT-LEAST-FREQUENT',
                        'WHERE-CAT-MOST-FREQUENT', 'WHERE-CAT-LEAST-FREQUENT',
                        'WHERE-NOT-MOST-FREQUENT', 'WHERE-NOT-LEAST-FREQUENT']
            self.append_sql_generated(sql_tags, queries, questions)

    def _generate_where_numerical(self, table_name: str, num_cols: list, df: pd.DataFrame):
        """
        Generates WHERE SQL queries and questions for numerical columns.
        Generates test for max, min, and mean values.

        Args:
            table_name (str): The name of the table in the database.
            num_cols (list): List of numerical columns.
            df (pd.DataFrame): The DataFrame containing the data.
        """

        def _generate_given_value(number, n_col):
            queries_n = [
                f'SELECT * FROM `{table_name}` WHERE `{n_col}` > {number}',
                f'SELECT * FROM `{table_name}` WHERE `{n_col}` < {number}',
            ]
            questions_n = [
                f'Show the data of the table "{table_name}" where "{n_col}" is greater than {number}',
                f'Show the data of the table "{table_name}" where "{n_col}" is less than {number}',
            ]
            return queries_n, questions_n

        if len(num_cols) == 0:
            return
        max_elements = [self._get_most_frequent_or_max_value(df[col].values)
                        for col in num_cols]
        min_elements = [self._get_least_frequent_or_min_value(df[col].values)
                        for col in num_cols]
        mean_values = [self._get_median_value(df[col].values) for col in num_cols]
        for col, max_value, min_value, mean_value in zip(num_cols, max_elements,
                                                         min_elements, mean_values):
            queries, questions = _generate_given_value(max_value, col)
            sql_tags = ['WHERE-NUM-MAX-VALUES-EMPTY', 'WHERE-NUM-MAX-VALUES']
            # avoid empty results
            self.append_sql_generated(sql_tags[1:], queries[1:], questions[1:])

            queries, questions = _generate_given_value(min_value, col)
            sql_tags = ['WHERE-NUM-MIN-VALUES', 'WHERE-NUM-MIN-VALUES-EMPTY']
            # avoid empty results
            self.append_sql_generated(sql_tags[:1], queries[:1], questions[:1])

            queries, questions = _generate_given_value(mean_value, col)
            sql_tags = ['WHERE-NUM-MEAN-VALUES'] * len(queries)
            self.append_sql_generated(sql_tags, queries, questions)

    @staticmethod
    def _get_most_frequent_or_max_value(values: np.array):
        """
        Returns the most frequent value if the input is categorical, or the maximum value if numerical.
        Null values are not considered in the calculation.

        Args:
            values (np.array): Array of values, either categorical or numerical.

        Returns:
            Union[None, Any]: Most frequent value if categorical, max value if numerical, or None if input is empty.
        """
        if len(values) == 0:
            return None
        values = values[~pd.isna(values)]
        # update the dtype after removing the null values
        values = np.array(values.tolist())
        if np.issubdtype(values.dtype, np.number):
            return np.max(values)
        else:
            unique_values, counts = np.unique(values, return_counts=True)
            index_of_max_count = np.argmax(counts)
            most_frequent_value: str = unique_values[index_of_max_count]
            return most_frequent_value.replace('"', '').replace("'", '').strip()

    @staticmethod
    def _get_least_frequent_or_min_value(values):
        """
        Returns the least frequent value if the input is categorical, or the minimum value if numerical.
        Null values are not considered in the calculation.

        Args:
            values (np.array): Array of values, either categorical or numerical.

        Returns:
            Union[None, Any]: Least frequent value if categorical, min value if numerical, or None if input is empty.
        """
        if len(values) == 0:
            return None
        values = values[~pd.isna(values)]
        # update the dtype after removing the null values
        values = np.array(values.tolist())
        if np.issubdtype(values.dtype, np.number):
            return np.min(values)
        else:
            unique_values, counts = np.unique(values, return_counts=True)
            index_of_min_count = np.argmin(counts)
            lest_frequent_value: str = unique_values[index_of_min_count]
            return lest_frequent_value.replace('"', '').replace("'", '').strip()

    @staticmethod
    def _get_median_value(values):
        """
        Returns the median value if the input is numerical. Null values are not considered in the calculation.

        Args:
            values (np.array): Array of numerical values.

        Returns:
            Union[None, float]: Mean value of the input array, or None if input is empty or non-numerical.
        """
        if len(values) == 0:
            return None
        values = values[~pd.isna(values)]
        if np.issubdtype(values.dtype, np.number):
            return np.mean(values)
        else:
            return None
