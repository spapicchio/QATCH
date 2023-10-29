import random
from abc import ABC, abstractmethod

import pandas as pd

from ..database_reader import SingleDatabase


class AbstractSqlGenerator(ABC):
    """
    An abstract base class for generating SQL tags, queries, and questions based on a database table.
    Subclasses must implement the abstract method `sql_generate`.

    :ivar SingleDatabase database: The SingleDatabase object representing the database to generate SQL from.
    :ivar int seed: The seed value for randomization. Default is 2023.
    :ivar dict sql_generated: A dictionary containing generated SQL tags, queries, and questions.
        Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def __init__(self, database: SingleDatabase, seed=2023):
        """
         Initialize the AbstractSqlGenerator object.
         :param SingleDatabase database: The SingleDatabase object representing the database to generate SQL from.
         :param int seed: The seed value for randomization. Default is 2023.
         """
        random.seed(seed)
        self.database = database
        self.sql_generated = {"sql_tags": [], "queries": [], "questions": []}

    def empty_sql_generated(self):
        """Resets the generated SQL tags, queries, and questions."""
        self.sql_generated = {"sql_tags": [], "queries": [], "questions": []}

    def append_sql_generated(self, sql_tags, queries, questions):
        """Appends generated SQL tags, queries, and questions to the existing ones."""
        assert len(sql_tags) == len(queries) == len(questions)
        self.sql_generated['sql_tags'].extend(sql_tags)
        self.sql_generated['queries'].extend(queries)
        self.sql_generated['questions'].extend(questions)

    @abstractmethod
    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Abstract method to generate SQL tags, queries, and questions based on the specified table.
        :param str table_name: The name of the table in the database.
        :return: A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}

        """
        raise NotImplementedError

    def _sample_cat_num_cols(self, table_name: str, sample=None) -> tuple[pd.DataFrame, list, list]:
        """
        Given the table name, returns the sampled categorical and numerical columns from the table.
        :param str table_name: The name of the table in the database.
        :param int sample: The number of columns to sample. Default is None.
        :return: A tuple containing the DataFrame, sampled categorical columns, and sampled numerical columns.
        """
        schema_df = self.database.get_schema_given(table_name)
        cat_attributes = schema_df.loc[schema_df.type == 'TEXT', 'name'].tolist()
        num_attributes = schema_df.loc[schema_df.type == 'INTEGER' or
                                       schema_df.type == 'REAL', 'name'].tolist()
        # TODO: current version store in memory the whole table, it could be a problem for big tables
        df = self.database.get_table_given(table_name)
        # substitute empty value with None
        df = df.replace(r'', None, regex=True)
        # avoid columns where all the values are None
        cat_cols = [col for col in cat_attributes if not all(df[col].isna())]
        num_cols = [col for col in num_attributes if not all(df[col].isna())]

        if sample is not None:
            # sample the categorical columns
            cat_cols = random.sample(cat_cols, sample) if len(cat_cols) >= sample else cat_cols
            num_cols = random.sample(num_cols, sample) if len(num_cols) >= sample else num_cols
        return df, cat_cols, num_cols

    @staticmethod
    def _get_col_comb_str(comb: list):
        """Given a combination of columns, return a string with the columns names"""
        return ", ".join([f'"{str(c)}"' for c in comb])

    @staticmethod
    def _comb_random(columns: list[str]) -> list[list[str]]:
        """Randomly select columns for each possible combinations between cols"""
        all_comb_num_cols = [num_cols for num_cols in range(1, len(columns) + 1)]
        return [random.sample(columns, k) for k in all_comb_num_cols]
