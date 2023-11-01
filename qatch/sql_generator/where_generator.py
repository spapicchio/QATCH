import numpy as np
import pandas as pd

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class WhereGenerator(AbstractSqlGenerator):
    """
    A class for generating WHERE SQL queries and corresponding questions based on a database table.

    :ivar SingleDatabase database: The SingleDatabase object representing the database to generate queries from.
    :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, questions, and results.
        Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str], "results": list[pd.DataFrame]}
    """

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Generates WHERE SQL queries and corresponding questions for both categorical and numerical columns.

        :param str table_name: The name of the table in the database.
        :return: A dictionary containing generated SQL tags, queries, questions, and results.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
        """
        self.empty_sql_generated()
        df, cat_cols, num_cols = self._sample_cat_num_cols(table_name)
        self.generate_where_categorical(table_name, cat_cols, df)
        self.generate_where_numerical(table_name, num_cols, df)
        return self.sql_generated

    def generate_where_categorical(self, table_name: str, cat_cols: list, df: pd.DataFrame):
        """
        Generates WHERE SQL queries and questions for categorical columns.
        Generates test for both most frequent and least frequent values.

        :param str table_name: The name of the table in the database.
        :param list cat_cols: List of categorical columns.
        :param pd.DataFrame df: The DataFrame containing the data.
        """
        if len(cat_cols) == 0:
            # no categorical attributes present
            return
        most_frequent_elements = [self.get_most_frequent_or_max_value(df[col].values) for col in cat_cols]
        least_frequent_elements = [self.get_least_frequent_or_min_value(df[col].values) for col in cat_cols]
        for col, most_freq, least_freq in zip(cat_cols, most_frequent_elements, least_frequent_elements):
            queries = [
                f"""SELECT * FROM "{table_name}" WHERE "{col}" == '{most_freq}'""",
                f"""SELECT * FROM "{table_name}" WHERE "{col}" == '{least_freq}'""",
                f"""SELECT * FROM "{table_name}" WHERE "{col}" != '{most_freq}'""",
                f"""SELECT * FROM "{table_name}" WHERE "{col}" != '{least_freq}'""",
                f"""SELECT * FROM "{table_name}" WHERE NOT "{col}" == '{most_freq}'""",
                f"""SELECT * FROM "{table_name}" WHERE NOT "{col}" == '{least_freq}'""",
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

    def generate_where_numerical(self, table_name: str, num_cols: list, df: pd.DataFrame):
        """
        Generates WHERE SQL queries and questions for numerical columns.
        Generates test for max, min, and mean values.

        :param str table_name: The name of the table in the database.
        :param list num_cols: List of numerical columns.
        :param pd.DataFrame df: The DataFrame containing the data.
        """

        def _generate_given_value(number, n_col):
            queries_n = [
                f'SELECT * FROM "{table_name}" WHERE "{n_col}" > "{number}"',
                f'SELECT * FROM "{table_name}" WHERE "{n_col}" < "{number}"',
            ]
            questions_n = [
                f'Show the data of the table "{table_name}" where "{n_col}" is greater than {number}',
                f'Show the data of the table "{table_name}" where "{n_col}" is less than {number}',
            ]
            return queries_n, questions_n

        if len(num_cols) == 0:
            return
        max_elements = [self.get_most_frequent_or_max_value(df[col].values)
                        for col in num_cols]
        min_elements = [self.get_least_frequent_or_min_value(df[col].values)
                        for col in num_cols]
        mean_values = [self.get_mean_value(df[col].values) for col in num_cols]
        for col, max_value, min_value, mean_value in zip(num_cols, max_elements,
                                                         min_elements, mean_values):
            queries, questions = _generate_given_value(max_value, col)
            sql_tags = ['WHERE-NUM-MAX-VALUES'] * len(queries)
            self.append_sql_generated(sql_tags, queries, questions)

            queries, questions = _generate_given_value(min_value, col)
            sql_tags = ['WHERE-NUM-MIN-VALUES'] * len(queries)
            self.append_sql_generated(sql_tags, queries, questions)

            queries, questions = _generate_given_value(mean_value, col)
            sql_tags = ['WHERE-NUM-MEAN-VALUES'] * len(queries)
            self.append_sql_generated(sql_tags, queries, questions)

    @staticmethod
    def get_most_frequent_or_max_value(values: np.array):
        """
        return most frequent value if categorical, max value if numerical.
        if Null value are present, they are not considered"""
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
            most_frequent_value = unique_values[index_of_max_count]
            return most_frequent_value.replace('"', '').replace("'", '').strip()

    @staticmethod
    def get_least_frequent_or_min_value(values):
        """
        return least frequent value if categorical, min value if numerical.
        if Null value are present, they are not considered"""
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
            lest_frequent_value = unique_values[index_of_min_count]
            return lest_frequent_value.replace('"', '').replace("'", '').strip()

    @staticmethod
    def get_mean_value(values):
        """
        return mean value if numerical.
        if Null value are present, they are not considered"""
        if len(values) == 0:
            return None
        values = values[~pd.isna(values)]
        if np.issubdtype(values.dtype, np.number):
            return np.mean(values)
        else:
            return None
