import logging

from src import TestGenerator
from .abstract_runner import Runner
from .utils import read_data


class RunnerProprietary(Runner):
    @property
    def test_generator(self):
        # 2. get proprietary tables
        tables = read_data(db_id=self.db_id,
                           model_name=self.name_model,
                           input_base_path_data=self.input_path_tables)

        # 3. inject null values in tables
        tables = self.inject_null_values_in_tables(tables)
        if self.verbose:
            logging.info(f'Tables stored in {self.save_database_tests}/{self.db_id}')

        # 4. init test generator
        test_generator = TestGenerator(
            db_save_path=self.save_database_tests,
            db_name=self.db_id,
            db_tables=tables
        )
        return test_generator

    def generate_tests(self):
        if self.verbose:
            logging.info('Starting generating tests')
        tables, tests_df = self.test_generator.generate(generators=self.sql_generators,
                                                        table_names=self.tbl_names,
                                                        save_spider_format=self.save_spider_format)

        return tables, tests_df.rename(columns={'query_result': self.query_result_col_name})
