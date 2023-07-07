import numpy as np
import pandas as pd

from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class WhereGenerator(AbstractSqlGenerator):

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str):
        """
        several WHERE conditions are tested:
            * WHERE column_name = value (only for categorical columns)
            * WHERE column_name != value
            * WHERE column_name > value
            * WHERE column_name < value
            * WHERE column_name >= value
            * WHERE column_name <= value
        """
        self.empty_sql_generated()
        self.generate_where_categorical(table_name)
        self.generate_where_numerical(table_name)
        return self.sql_generated

    def generate_where_categorical(self, table_name: str):
        df, cat_cols, _ = self._get_df_cat_num_cols(table_name)
        if not cat_cols:
            return

        most_frequent_elements = [self.get_most_frequent_or_max_value(df[col].values) for col in cat_cols]
        least_frequent_elements = [self.get_least_frequent_or_min_value(df[col].values) for col in cat_cols]

        for col, most_freq, least_freq in zip(cat_cols, most_frequent_elements, least_frequent_elements):
            queries = [
                f'SELECT * FROM "{table_name}" WHERE "{col}" == "{most_freq}"',
                f'SELECT * FROM "{table_name}" WHERE "{col}" == "{least_freq}"',
                f'SELECT * FROM "{table_name}" WHERE "{col}" != "{most_freq}"',
                f'SELECT * FROM "{table_name}" WHERE "{col}" != "{least_freq}"',
            ]

            questions = [
                f'Show the data of the table "{table_name}" where "{col}" is equal to {most_freq}',
                f'Show the data of the table "{table_name}" where "{col}" is equal to {least_freq}',
                f'Show the data of the table "{table_name}" where "{col}" is different from {most_freq}',
                f'Show the data of the table "{table_name}" where "{col}" is different from {least_freq}'
            ]

            results = [self.database.run_query(query) for query in queries]

            sql_tags = ['WHERE-CAT-MOST-FREQUENT', 'WHERE-CAT-LEAST-FREQUENT',
                        'WHERE-CAT-MOST-FREQUENT', 'WHERE-CAT-LEAST-FREQUENT']
            self.append_sql_generated(sql_tags, queries, questions, results)

        return self.sql_generated

    @staticmethod
    def get_most_frequent_or_max_value(values: np.array):
        """
        return most frequent value if categorical, max value if numerical
        if Null value are present, they are not considered"""
        if len(values) == 0:
            return None
        values = values[~pd.isna(values)]
        if np.issubdtype(values.dtype, np.number):
            return np.max(values)
        else:
            unique_values, counts = np.unique(values, return_counts=True)
            index_of_max_count = np.argmax(counts)
            most_frequent_value = unique_values[index_of_max_count]
            return most_frequent_value

    @staticmethod
    def get_least_frequent_or_min_value(values):
        """
        return least frequent value if categorical, min value if numerical
        if Null value are present, they are not considered"""
        if len(values) == 0:
            return None
        values = values[~pd.isna(values)]
        if np.issubdtype(values.dtype, np.number):
            return np.min(values)
        else:
            unique_values, counts = np.unique(values, return_counts=True)
            index_of_min_count = np.argmin(counts)
            lest_frequent_value = unique_values[index_of_min_count]
            return lest_frequent_value

    @staticmethod
    def get_mean_value(values):
        """
        return mean value if numerical
        if Null value are present, they are not considered"""
        if len(values) == 0:
            return None
        values = values[~pd.isna(values)]
        if np.issubdtype(values.dtype, np.number):
            return np.mean(values)
        else:
            return None

    def generate_where_numerical(self, table_name):
        def _generate_given_value(number, n_col):
            queries_n = [
                f'SELECT * FROM "{table_name}" WHERE "{n_col}" > "{number}"',
                f'SELECT * FROM "{table_name}" WHERE "{n_col}" < "{number}"',
                f'SELECT * FROM "{table_name}" WHERE "{n_col}" >= "{number}"',
                f'SELECT * FROM "{table_name}" WHERE "{n_col}" <= "{number}"',
            ]
            questions_n = [
                f'Show the data of the table "{table_name}" where "{n_col}" is greater than {number}',
                f'Show the data of the table "{table_name}" where "{n_col}" is less than {number}',
                f'Show the data of the table "{table_name}" where "{n_col}" is greater than or equal to {number}',
                f'Show the data of the table "{table_name}" where "{n_col}" is less than or equal to {number}',
            ]
            return queries_n, questions_n

        df, _, num_cols = self._get_df_cat_num_cols(table_name)
        if not num_cols:
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
            results = [self.database.run_query(query) for query in queries]
            self.append_sql_generated(sql_tags, queries, questions, results)

            queries, questions = _generate_given_value(min_value, col)
            sql_tags = ['WHERE-NUM-MIN-VALUES'] * len(queries)
            results = [self.database.run_query(query) for query in queries]
            self.append_sql_generated(sql_tags, queries, questions, results)

            queries, questions = _generate_given_value(mean_value, col)
            sql_tags = ['WHERE-NUM-MEAN-VALUES'] * len(queries)
            results = [self.database.run_query(query) for query in queries]
            self.append_sql_generated(sql_tags, queries, questions, results)

        return self.sql_generated

