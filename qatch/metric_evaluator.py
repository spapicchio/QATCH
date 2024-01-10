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
    Class for evaluating SQL query prediction metrics using target results and predicted outputs.

    Attributes:
        databases (MultipleDatabases): Object representing database connections.
            This attribute stores information about multiple database connections.

        metrics (list[str]): List of metric names to be evaluated. Default metrics include:
            ['cell_precision', 'cell_recall', 'tuple_cardinality', 'tuple_constraint', 'tuple_order']
    """

    def __init__(self, databases: MultipleDatabases, metrics: list[str] | str | None = None):
        """
        initialize the MetricEvaluator object.

        Args:
            databases (MultipleDatabases): Object representing database connections.
                This attribute stores information about multiple database connections.
            metrics (list[str] | str | None): List of metric names to be evaluated. Default metrics include:
                ['cell_precision', 'cell_recall', 'tuple_cardinality', 'tuple_constraint', 'tuple_order']
        """
        if metrics is None:
            metrics = ['cell_precision', 'cell_recall',
                       'tuple_cardinality', 'tuple_constraint',
                       'tuple_order']

        self.metrics = metrics if isinstance(metrics, list) else [metrics]

        self._tags_generator = {
            'cell_precision': CellPrecisionTag,
            'cell_recall': CellRecallTag,
            'tuple_cardinality': TupleCardinalityTag,
            'tuple_constraint': TupleConstraintTag,
            'tuple_order': TupleOrderTag,
        }
        self.databases = databases

    def evaluate_with_df(self, df, prediction_col_name: str, task: str, keep_target: bool = False
                         ) -> pd.DataFrame:
        """
        Evaluate the specified metrics using the provided DataFrame containing query results and predictions.
        The df must contain 'query' and 'db_id' columns, and the "prediction_col_name" must be present in the df.
        For SP task, it generates the prediction results for the queries that are not equal to the target.

        Args:
            df (pd.DataFrame): The DataFrame containing query results, prediction, and other relevant columns.
            prediction_col_name (str): The name of the column containing prediction values.
            task (str): The task type, either 'SP' for SQL prediction or other task types.
            keep_target (bool): Whether to keep the target column in the output DataFrame. Default is False.

        Returns:
            pd.DataFrame: A DataFrame with added metric columns.

        Examples:
            Given a DataFrame "tests_df" with the following columns: ['query', 'db_id', 'pred_tapas']
            and a MultipleDatabases object connected with the db_id,
            >>> evaluator = MetricEvaluator(databases)
            >>> tests_df = evaluator.evaluate_with_df(tests_df, 'pred_tapas', 'SP')
            >>> tests_df.columns.tolist()
            [query, db_id, pred_tapas, cell_precision_pred_tapas, cell_recall_pred_tapas, tuple_cardinality_pred_tapas, tuple_constraint_pred_tapas, tuple_order_pred_tapas]
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
        mask_equal = df[prediction_col_name] == 'EQUAL'

        for metric in self.metrics:
            generator = self._tags_generator[metric]()
            # initialize the metric column
            metric_col_name = f'{metric}_{prediction_col_name}'
            df.loc[:, metric_col_name] = None
            if metric == 'tuple_order':
                # when the target and prediction are equal, the metric is 1
                df.loc[mask_order & mask_equal, metric_col_name] = 1
                mask = mask_order & ~mask_equal
            else:
                # when the target and prediction are equal, the metric is 1
                df.loc[mask_equal, metric_col_name] = 1
                mask = ~mask_equal

            # evaluate the metric only for the test where the prediction is not equal to the target
            tqdm.pandas(desc=f'Evaluating {metric_col_name}')
            df.loc[mask, metric_col_name] = df[mask].progress_apply(
                lambda r: generator.evaluate_single_test_metric(r[target_col_name], r[prediction_col_name]),
                axis=1)
        # at the end drop the columns that are not needed anymore
        if not keep_target:
            df = df.drop(columns=[target_col_name, prediction_col_name]) if task == 'SP' \
                else df.drop(columns=[target_col_name])
        return df

    def _get_query_results_from_db(self, df) -> tuple[pd.DataFrame, str]:
        """
        Retrieve query results for the "query" column.
        Since these queries represent the target values, this function raises an error if the query is not valid.

        Args:
            df (pd.DataFrame): The input DataFrame containing query information.

        Returns:
            tuple[pd.DataFrame, str]: A tuple containing the new DataFrame with query results and the column name for query results.

        Raises:
            sqlite3.OperationalError: If the query is not valid (the target must be correct).
        """
        query_column = 'query'
        # group-by the df for each db_id present
        grouped_by_db_df = df.groupby('db_id').agg(list)
        # for each db_id get the results of the query from the db
        tqdm.pandas(desc='Getting target results')
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

        Args:
            target (str): The target string.
            prediction (str): The prediction string.

        Returns:
            bool: True if cleaned prediction equals cleaned target, False otherwise.
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

        Args:
            df (pd.DataFrame): The input DataFrame containing query information.
            prediction_col_name (str): The name of the column containing prediction values.

        Returns:
            tuple[pd.DataFrame, str]: A tuple containing the new DataFrame with query results and the column name for query results.
        """

        def wrapper_prediction(db_id, query):
            """in case the prediction return an error, we return None"""
            # to avoid multiple queries in the same string error
            query = query.replace(';', '')
            try:
                output = self.databases.run_query(db_id, query)
            except sqlite3.Error as e:
                # catch any possible error of prediction and return None
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
