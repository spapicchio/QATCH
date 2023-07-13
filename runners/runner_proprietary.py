import logging

from tqdm import tqdm

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
            db_save_path=f'{self.save_database_tests}/{self.db_id}',
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

    def predict(self, tables, tests_df):
        if self.verbose:
            logging.info('Starting predictions')
        tqdm.pandas(desc=f'Predictions for {self.name_model}')
        tests_df[self.prediction_col_name] = tests_df.progress_apply(
            lambda row: self.model.predict(table=tables[row.tbl_name],
                                           queries=row.question,
                                           tbl_name=row.tbl_name),
            axis=1
        )
        return tests_df

    def evaluate(self, tests_df):
        if self.verbose:
            logging.info('Starting evaluating tests')

        tests_df = self.sp_predictions2query_result(tests_df, is_spider=False)
        tests_df = self.metric_evaluator.evaluate_with_df(tests_df,
                                                          target=self.query_result_col_name,
                                                          predictions=self.prediction_col_name,
                                                          task=self.task)
        self.change_col_name = False

        return tests_df
