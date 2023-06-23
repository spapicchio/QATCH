import os.path
from typing import Literal

import pandas as pd
from tqdm import tqdm

from .abstract_test_generator import AbstractTestGenerator
from .database_reader.single_database import SingleDatabase
from .sql_generator import (SelectGenerator, DistinctGenerator,
                            OrderByGenerator, WhereGenerator,
                            GroupByGenerator, HavingGenerator)


class TestGenerator(AbstractTestGenerator):
    def __init__(self,
                 db_save_path: str,
                 db_name: str,
                 db_tables: dict[str, pd.DataFrame],
                 seed=2023):
        self.db_save_path = db_save_path
        if not os.path.exists(self.db_save_path):
            # create directory
            os.makedirs(self.db_save_path)

        self.database = SingleDatabase(db_path=db_save_path, db_name=db_name, tables=db_tables)
        self.seed = seed
        self.generators = {'select': SelectGenerator,
                           'orderby': OrderByGenerator,
                           'distinct': DistinctGenerator,
                           'where': WhereGenerator,
                           'groupby': GroupByGenerator,
                           'having': HavingGenerator}

    def generate(self,
                 generators: Literal['select', 'orderby', 'distinct', 'where'] | list[str] | None = None,
                 table_names: str | list[str] | None = None,
                 save_spider_format: bool = False
                 ) -> tuple[
        dict[str, pd.DataFrame],
        pd.DataFrame
    ]:
        generators, table_names = self._init_parameters(generators, table_names)
        tests_df = pd.DataFrame()
        for generator in tqdm(generators, desc="Generating tests"):
            for table_name in table_names:
                sql_generated = generator.sql_generate(table_name)
                sql_tags = sql_generated['sql_tags']
                queries = sql_generated['queries']
                questions = sql_generated['questions']
                results = sql_generated['results']
                df = self._build_df(table_name, sql_tags, queries, questions, results)
                tests_df = pd.concat([tests_df, df])
        if save_spider_format:
            self._save_spider_format(tests_df)

        return self.database.tables, tests_df

    def _init_parameters(self, generators, table_names) -> tuple[list, list]:
        if table_names is None:
            # if not specified, use all tables
            table_names = list(self.database.tables.keys())

        if generators is None:
            generators = list(self.generators.keys())
        if isinstance(table_names, str):
            table_names = [table_names]
        if isinstance(generators, str):
            generators = [generators]

        for name in table_names:
            if name not in self.database.tables:
                raise ValueError(f"Table {name} not found in database."
                                 f" Provided tables are "
                                 f"{list(self.database.tables.keys())}")

        generators = [self.generators[gen_type](self.database, self.seed)
                      for gen_type in generators]

        return generators, table_names

    def _build_df(self, table_name, sql_tags, queries, questions, results):
        return pd.DataFrame({
            'db_id': [self.database.db_name] * len(sql_tags),
            'tbl_name': [table_name] * len(sql_tags),
            'sql_tags': sql_tags,
            'query': queries,
            'question': questions,
            'query_result': results})

    def _save_spider_format(self, df):
        path = os.path.join(self.db_save_path, 'tests.json')
        df.to_json(path, orient='records')

        table_names = df.tbl_name.unique().tolist()
        column_names = []
        column_types = []

        # for each table
        for table_id, name in enumerate(table_names):
            table = self.database.get_table_from_name(name)
            table = table.infer_objects()
            columns = table.columns.values.tolist()
            # for each column save name and type
            for col_i, type_col in enumerate(table.dtypes):
                type_col = 'number' if type_col == float or type_col == int else 'text'
                column_names.append([table_id, columns[col_i]])
                column_types.append(type_col)

        # create SPIDER dataframe for tables.json
        tables = pd.DataFrame({
            'column_names': [column_names],
            'column_names_original': [column_names],
            'column_types': [column_types],
            'db_id': self.database.db_name,
            'foreign_keys': [[]],
            'primary_keys': [[]],
            'table_names': [table_names],
            'table_names_original': [table_names],
        })

        path = os.path.join(self.db_save_path, 'tables.json')
        tables.to_json(path, orient='records')
