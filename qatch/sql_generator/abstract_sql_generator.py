import random
from abc import ABC, abstractmethod

import pandas as pd

from ..database_reader import SingleDatabase


class AbstractSqlGenerator(ABC):
    """An abstract base class for generating SQL tags, queries, and questions based on a database table.
    Subclasses must implement the abstract method `sql_generate`.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate SQL from.
        seed (int): The seed value for randomization. Default is 2023.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def __init__(self, database: SingleDatabase, seed=2023):
        self.seed = seed
        self.database = database
        self.sql_generated = {"sql_tags": [], "queries": [], "questions": []}

    def empty_sql_generated(self):
        """Resets the generated SQL tags, queries, and questions."""
        self.sql_generated = {"sql_tags": [], "queries": [], "questions": []}

    def append_sql_generated(self, sql_tags, queries, questions):
        """Appends generated SQL tags, queries, and questions to the existing ones.

        Args:
            sql_tags (list[str]): List of generated SQL tags.
            queries (list[str]): List of generated SQL queries.
            questions (list[str]): List of generated questions.
        """
        assert len(sql_tags) == len(queries) == len(questions)
        self.sql_generated['sql_tags'].extend(sql_tags)
        self.sql_generated['queries'].extend(queries)
        self.sql_generated['questions'].extend(questions)

    @abstractmethod
    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        Abstract method to generate SQL tags, queries, and questions based on the specified table.

        Args:
            table_name (str): The name of the table in the database.

        Returns:
            dict[str, list]: A dictionary containing generated SQL tags, queries, and questions.
                Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
        """
        raise NotImplementedError

    def _sample_cat_num_cols(self, table_name: str, sample=None) -> tuple[pd.DataFrame, list, list]:
        """
        Given the table name, returns the sampled categorical and numerical columns from the table.

        Args:
            table_name (str): The name of the table in the database.
            sample (int): The number of columns to sample. Default is None.

        Returns:
            tuple[pd.DataFrame, list, list]: A tuple containing the DataFrame, sampled categorical columns, and
            sampled numerical columns.
        """
        schema_df = self.database.get_schema_given(table_name)
        cat_attributes = schema_df.loc[schema_df.type == 'TEXT', 'name'].tolist()
        num_attributes = schema_df.loc[(schema_df.type == 'INTEGER') |
                                       (schema_df.type == 'REAL'), 'name'].tolist()
        # TODO: current version store in memory the whole table, it could be a problem for big tables
        df = self.database.get_table_given(table_name)
        # substitute empty value with None
        df = df.replace(r'', None, regex=True)
        # avoid columns where all the values are None
        cat_cols = [col for col in cat_attributes if not all(df[col].isna())]
        num_cols = [col for col in num_attributes if not all(df[col].isna())]

        if sample is not None:
            # sample the categorical columns
            random.seed(self.seed)
            cat_cols = random.sample(cat_cols, sample) if len(cat_cols) >= sample else cat_cols
            random.seed(self.seed)
            num_cols = random.sample(num_cols, sample) if len(num_cols) >= sample else num_cols
        return df, cat_cols, num_cols

    @staticmethod
    def _get_col_comb_str(comb: list) -> str:
        """Given a combination of columns, return a string with the columns names.

        Args:
            comb (list): List of column names.

        Returns:
            str: String representation of the column combination.
        """
        return ", ".join([f'`{str(c)}`' for c in comb])

    def _comb_random(self, columns: list[str]) -> list[list[str]]:
        def _get_sample(_columns, k):
            random.seed(self.seed)
            return random.sample(_columns, k)

        """Randomly select columns for each possible combinations between columns.

        Args:
            columns (list): List of column names.

        Returns:
            list[list[str]]: List of randomly selected column combinations.
        """
        all_comb_num_cols = [num_cols for num_cols in range(1, len(columns) + 1)]
        return [_get_sample(columns, k) for k in all_comb_num_cols]
