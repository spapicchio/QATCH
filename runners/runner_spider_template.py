import pandas as pd

from src import TestGenerator
from .abstract_runner import Runner
from .utils import get_spider_table_paths, random_db_id_spider_tables, transform_spider_tables_key


class RunnerSpiderTemplate(Runner):
    @property
    def test_generator(self):
        # 1: get spider tables
        tables = get_spider_table_paths(self.pickle_spider_tables, self.input_path_tables)
        # 2 transform tables key
        tables = transform_spider_tables_key(tables)
        # 3. select 10 random db_ids
        tables = random_db_id_spider_tables(tables, self.seed, k=self.sample_k_spider_table)
        # 4 inject null values
        tables = self.inject_null_values_in_spider_tables(tables)
        test_generators = []
        for db_id in tables:
            test_generator = TestGenerator(
                db_save_path=self.save_database_tests,
                db_name=db_id,
                db_tables=tables[db_id]
            )
            test_generators.append(test_generator)
        return test_generators

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pickle_spider_tables = kwargs['pickle_spider_tables']
        self.sample_k_spider_table = kwargs['sample_k_spider_table']

    def generate_tests(self):
        dfs = []
        final_tables = {}
        for test_generator in self.test_generator:
            tables, tests_df = test_generator.generate(self.sql_generators,
                                                       save_spider_format=self.save_spider_format)
            dfs.append(tests_df.reset_index(drop=True))
            final_tables.update(tables)
        tests_df = pd.concat(dfs).reset_index(drop=True)
        return final_tables, tests_df.rename(columns={'query_result': self.query_result_col_name})

    def inject_null_values_in_spider_tables(self, tables: dict[str, dict[str, pd.DataFrame]]):
        new_tables = {}
        if self.inject_null_percentage > 0.0:
            for key, df_tables in tables.items():
                db_tables = self.inject_null_values_in_tables(df_tables)
                new_tables[key] = db_tables
        else:
            new_tables = tables  # no null values
        return new_tables
