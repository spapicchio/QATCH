import json
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import Tapas, Tapex, ChatGPT, MetricEvaluator
from utils import get_predictions_results_from_dbs


class Runner(ABC):
    def __init__(self, **kwargs):
        self.change_col_name = False
        self.model_name_path = kwargs['model_name_path']
        self.task = kwargs['task']
        self.db_id = kwargs['db_id']
        self.input_path_tables = kwargs['input_path_tables']
        self.save_database_tests = kwargs['save_database_tests']
        self.verbose = kwargs['verbose']
        self.metrics = kwargs['metrics']
        self.percentage = kwargs['percentage']
        self.seed = kwargs['seed']
        self.sql_generators = kwargs['sql_generators']
        self.save_spider_format = kwargs['save_spider_format']
        self.tbl_names = kwargs['tbl_names']
        if self.task == 'SP' and self.model_name_path != 'chatgpt':
            raise ValueError('QATCH current version only supports chatgpt for SP task')

    @property
    def prediction_col_name(self):
        if self.change_col_name:
            return f'query_result_predictions_{self.name_model}'
        return f'prediction_{self.name_model}'

    @property
    def query_result_col_name(self):
        return f'query_result_{self.name_model}'

    @property
    def name_model(self):
        name = None
        name = 'tapas' if 'tapas' in self.model_name_path else name
        name = 'tapex' if 'tapex' in self.model_name_path else name
        name = 'resdsql' if 'resdsql' in self.model_name_path else name
        name = 'chatgpt' if 'chatgpt' in self.model_name_path else name
        if name is None:
            raise KeyError('Accepted model are tapas, tapex, chatgpt')
        return name

    @property
    @abstractmethod
    def test_generator(self):
        raise NotImplementedError

    @property
    def metric_evaluator(self):
        return MetricEvaluator(metrics=self.metrics)

    @property
    def model(self):
        name2model = {'tapas': Tapas, 'tapex': Tapex, 'chatgpt': ChatGPT}
        if self.name_model == 'chatgpt':
            with open('credentials.json', 'r') as f:
                credentials = json.load(f)
            model = ChatGPT(api_key=credentials['api_key_chatgpt'],
                            api_org=credentials['api_org_chatgpt'],
                            test_type=self.task)
        else:
            model = name2model[self.name_model](self.model_name_path)
        return model

    @abstractmethod
    def generate_tests(self):
        raise NotImplementedError

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

        tests_df = self.sp_predictions2query_result(tests_df)
        tests_df = self.metric_evaluator.evaluate_with_df(tests_df,
                                                          target=self.query_result_col_name,
                                                          predictions=self.prediction_col_name,
                                                          task=self.task)
        self.change_col_name = False

        return tests_df

    def inject_null_values_in_tables(self, tables: dict[str, pd.DataFrame]):
        """inject null into the tables"""
        np.random.seed(self.seed)
        if self.percentage > 0.0:
            tables = {key: df.mask(np.random.random(df.shape) < self.percentage)
                      for key, df in tables.items()}
        return tables

    def sp_predictions2query_result(self, tests_df: pd.DataFrame):
        if self.task == 'SP':
            tests_df = get_predictions_results_from_dbs(base_path_db=self.save_database_tests,
                                                        df=tests_df,
                                                        predictions=self.prediction_col_name)
            self.change_col_name = True
            tests_df.rename(columns={'query_result_predictions': self.prediction_col_name},
                            inplace=True)

        return tests_df
