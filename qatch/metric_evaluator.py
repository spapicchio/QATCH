import logging
import sqlite3

import pandas as pd
from tqdm import tqdm

from .database_reader import MultipleDatabases
from .metrics.cell_precision_tag import CellPrecisionTag
from .metrics.cell_recall_tag import CellRecallTag
from .metrics.tuple_cardinality_tag import TupleCardinalityTag
from .metrics.tuple_constraint_tag import TupleConstraintTag
from .metrics.tuple_order_tag import TupleOrderTag


class MetricEvaluator:
    """
    Use this class to have an interface for the evaluation of the metrics


    :ivar MultipleDatabases databases: The MultipleDatabases object representing the database connections.
    :ivar metrics: A list of metric names to be evaluated. Default metrics include:
                   ['cell_precision', 'cell_recall', 'tuple_cardinality', 'tuple_constraint', 'tuple_order']
    :ivar dict tags_generator: A dictionary mapping metric names to corresponding metric tag generator classes.
    """

    def __init__(self, databases: MultipleDatabases, metrics: list[str] | str | None = None):
        if metrics is None:
            metrics = ['cell_precision', 'cell_recall',
                       'tuple_cardinality', 'tuple_constraint',
                       'tuple_order']

        self.metrics = metrics if isinstance(metrics, list) else [metrics]

        self.tags_generator = {
            'cell_precision': CellPrecisionTag,
            'cell_recall': CellRecallTag,
            'tuple_cardinality': TupleCardinalityTag,
            'tuple_constraint': TupleConstraintTag,
            'tuple_order': TupleOrderTag,
        }
        self.databases = databases

    def evaluate_with_df(self, df, prediction_col_name: str, task: str):
        """
        Evaluate the specified metrics using the provided DataFrame containing query results and predictions.
        The df must contain 'query' and 'db_id' columns, and the "prediction_col_name" must be present in the df.
        For SP task, it generates the prediction results for the queries that are not equal to the target.

        :param df: The DataFrame containing query results, prediction, and other relevant columns.
        :param prediction_col_name: The name of the column containing prediction values.
        :param task: The task type, either 'SP' for SQL prediction or other task types.
        :return: A DataFrame with added metric columns.
        """
        # get target values
        df, target_col_name = self._get_query_results_from_db(df)
        if task == 'SP':
            # get prediction values for the SP task and the new prediction_col_name
            df, prediction_col_name = self._get_SP_query_results_from_db(df, prediction_col_name)

        # only the test where order is present
        queries = df['query'].str.lower()
        mask_order = queries.str.contains('order')

        # create the mask for the EQUAL case (no need to run the evaluation)
        mask_equal = df[prediction_col_name] == 'EQUAL' if task == 'SP' else [False] * len(df)

        for metric in self.metrics:
            generator = self.tags_generator[metric]()
            # initialize the metric column
            df.loc[:, metric] = None
            if metric == 'tuple_order':
                # when the target and prediction are equal, the metric is 1
                df.loc[mask_order & mask_equal, metric] = 1
                mask = mask_order & ~mask_equal
            else:
                # when the target and prediction are equal, the metric is 1
                df.loc[mask_equal, metric] = 1
                mask = ~mask_equal

            # evaluate the metric only for the test where the prediction is not equal to the target
            tqdm.pandas(desc=f'Evaluating {metric}')
            df.loc[mask, metric] = df[mask].progress_apply(
                lambda r: generator.evaluate_single_test_metric(r[target_col_name], r[prediction_col_name]),
                axis=1)
        # at the end drop the columns that are not needed anymore
        df = df.drop(columns=[target_col_name, prediction_col_name]) if task == 'SP' \
            else df.drop(columns=[target_col_name])
        return df

    def _get_query_results_from_db(self, df) -> tuple[pd.DataFrame, str]:
        """
        Retrieve query results for the "query" column.
        Since these queries represent the target values, this function raises an error if the query is not valid.

        :param pd.DataFrame df: The input DataFrame containing query information.
        :return: A tuple containing the new DataFrame with query results and the column name for query results.
        :raise: a sqlite3.OperationalError error if the query is not valid (the target must be correct)
        """
        query_column = 'query'
        # group-by the df for each db_id present
        grouped_by_db_df = df.groupby('db_id').agg(list)
        # for each db_id get the results of the query from the db
        tqdm.pandas(desc=f'Getting {query_column} result')
        grouped_by_db_df[f'{query_column}_result'] = grouped_by_db_df.progress_apply(
            lambda row: self.databases.run_multiple_queries(row.name, row[query_column]),
            axis=1
        )
        # expand the grouped df
        columns = grouped_by_db_df.columns.tolist()
        df = grouped_by_db_df.explode(columns).reset_index()
        return df, f'{query_column}_result'

    @staticmethod
    def _create_mask_target_equal_prediction(target: str, prediction: str) -> bool:
        """
        Create a mask based on whether the target and prediction strings are equal after cleaning.

        :param str target: The target string.
        :param str prediction: The prediction string.
        :return: True if cleaned prediction equals cleaned target, False otherwise.
        """
        new_target = (target.lower()
                      .replace(" ,", ",").replace("  ", " ").replace('"', '').replace("'", '')
                      .strip())

        new_pred = (prediction.lower()
                    .replace(" ,", ",").replace("  ", " ").replace('"', '').replace("'", '')
                    .replace(' ( ', '(').replace(' )', ')')
                    .strip())
        return True if new_pred == new_target else False

    def _get_SP_query_results_from_db(self, df: pd.DataFrame, prediction_col_name: str
                                      ) -> tuple[pd.DataFrame, str]:
        """
        Retrieve query results for the "prediction_col_name" column.
        Since this is the prediction of SP model, this function returns None if the query is not valid.

        :param pd.DataFrame df: The input DataFrame containing query information.
        :param str prediction_col_name: The name of the column containing prediction values.
        :return: tuple containing the new DataFrame with query results and the column name for query results.
        """

        def wrapper_prediction(db_id, query):
            """in case the prediction return an error, we return None"""
            try:
                output = self.databases.run_query(db_id, query)
            except sqlite3.OperationalError as e:
                logging.error(e)
                return None
            else:
                return output

        mask_equal = df.apply(
            lambda row: self._create_mask_target_equal_prediction(row['query'],
                                                                  row[prediction_col_name]),
            axis=1
        )
        # create the new column for the prediction
        new_prediction_col_name = f'{prediction_col_name}_result'
        df.loc[mask_equal, new_prediction_col_name] = 'EQUAL'
        # get prediction values for the elements not equal
        tqdm.pandas(desc='Getting prediction results for not equal SQL predictions')
        df.loc[~mask_equal, new_prediction_col_name] = df.loc[~mask_equal].progress_apply(
            lambda row: wrapper_prediction(row['db_id'], row[prediction_col_name]),
            axis=1
        )
        return df, new_prediction_col_name
