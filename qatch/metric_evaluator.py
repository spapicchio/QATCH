import logging
import sqlite3

import numpy as np
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
        if metrics is None:
            metrics = ['cell_precision', 'cell_recall',
                       'tuple_cardinality', 'tuple_constraint',
                       'tuple_order']

        self.metrics = metrics if isinstance(metrics, list) else [metrics]

        self._tags_generator = {
            'cell_precision': CellPrecisionTag(),
            'cell_recall': CellRecallTag(),
            'tuple_cardinality': TupleCardinalityTag(),
            'tuple_constraint': TupleConstraintTag(),
            'tuple_order': TupleOrderTag(),
        }
        self.databases = databases

    def evaluate_with_df(self, df, prediction_col_name: str, task: str, target_col_name: str = 'query',
                         keep_target: bool = False
                         ) -> pd.DataFrame:
        """Evaluates SQL queries for various metrics including cell precision, cell recall,
        tuple cardinality, tuple constraint, tuple order.

        For each row in the input DataFrame, it evaluates either the task as a QA
        (Question Answering) or SP (Semantic Parsing). Then, it concatenates the original DataFrame
        and the evaluated metric DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame where each row represents a test.
            prediction_col_name (str): Name of the column in the DataFrame that contains predictions.
            task (str): Type of evaluation task. Could be `QA` or `SP`.
            target_col_name (str): Name of the column in the DataFrame that contains target queries.
            Default is `'query'`.
            keep_target (bool): FALSE by default. If TRUE, keeps the target query.

        Returns:
            pd.DataFrame: Output DataFrame that has the original DataFrame along with the evaluated metric DataFrame.
        """
        if task.upper() == 'QA':
            # add the new metrics at the bottom of the dataframe
            df_metrics = df.apply(lambda row: self.evaluate_single_test_QA(row.to_dict(),
                                                                           prediction_col_name,
                                                                           target_col_name),
                                  axis=1, result_type='expand')
        else:
            df_metrics = df.apply(lambda row: self.evaluate_single_test_SP(row.to_dict(),
                                                                           prediction_col_name,
                                                                           target_col_name), axis=1,
                                  result_type='expand')
        return pd.concat([df, df_metrics], axis=1).replace({np.nan: None})

    def evaluate_single_test_QA(self, test: dict, prediction_col_name: str, target_col_name: str) -> dict:
        """
        Evaluates metric scores on a single test QA task where a test is a dictionary (or pd.Series) and the
        `prediction_col_name` and `target_col_name` are the column names in the test data containing model predictions
        and actual target values respectively.

        Args:
            test (dict | pd.Series): A dictionary or pandas Series containing a single test data. The keys (columns for Series)
                should include `prediction_col_name` and `target_col_name`.
            prediction_col_name (str): String representing the key in `test` dictionary (or column in `test` pandas Series)
                where the predicted values are.
            target_col_name (str): String representing the key in `test` dictionary (or column in `test` pandas Series)
                where the actual target values are.

        Returns:
            dict: A dictionary with keys are metric name and value is the evaluated metric score for each metric in `self.metrics`.

        Examples:
            >>> eval_task = MetricEvaluator(databases, metrics=['cell_precision', 'cell_recall'])
            >>> test = {"sql_tags": "SELECT",
            ...         "prediction": [["wales", "scotland"], ["england"]],
            ...         "target": [["scotland", "wales"], ["england"]]}
            >>> prediction_col_name = "prediction"
            >>> target_col_name = "target"
            >>>result = eval_task.evaluate_single_test_QA(test, prediction_col_name, target_col_name)
            >>> print(result)
            {'cell_precision_prediction': 0.66, 'cell_recall_prediction': 1.0}

        Notes:
            In the above example, `cell_precision_prediction` is calculated as 2/3 = 0.66,
            "wales" and "scotland" are in the correct position.
            `cell_recall_prediction` is 2/2 = 1, all actual outcomes are included in the prediction.
        """

        # Runs the target query on the database
        new_target_col = f'{target_col_name}_result'
        try:
            test[new_target_col] = self.databases.run_query(test['db_id'], test[target_col_name])
        except sqlite3.Error as e:
            # catch any possible error of prediction and return all zeros
            logging.error(e)
            return {f'{metric}_{prediction_col_name}': 0 for metric in self.metrics}

        metric2evaluation = {f'{metric}_{prediction_col_name}': None for metric in self.metrics}
        for metric in self.metrics:
            generator = self._tags_generator[metric]
            # initialize the metric column
            # evaluate the metric only for the test where the prediction is not equal to the target
            tqdm.pandas(desc=f'Evaluating {metric}')
            if metric == 'tuple_order' and 'order by' not in test[target_col_name].lower():
                continue
            evaluation = generator.evaluate_single_test_metric(test[new_target_col], test[prediction_col_name])
            metric2evaluation[f'{metric}_{prediction_col_name}'] = evaluation
        return metric2evaluation

    def evaluate_single_test_SP(self, test: dict, prediction_col_name: str, target_col_name: str) -> dict:
        """
        Evaluates metrics for a single SQL prediction test by fetching the results of the predicted and
        target queries from the database.

        This function fetches results based on provided `prediction_col_name` and `target_col_name`. Then it evaluates
        performance of the prediction by invoking `evaluate_single_test_QA`.

        Args:
            self (MetricEvaluator): The object instance the method is called on.
            test (dict | pd.Series): The test data as a dictionary or pandas Series. It contains the 'db_id' (database identifier).
                                     It is expected to have 'predictions_SP' and 'target_SP' keys/columns updated in process.
            prediction_col_name (str): The name of column where prediction is stored.
            target_col_name (str): The name of column where the target is stored.

        Returns:
            dict: A dictionary containing evaluation results obtained from `evaluate_single_test_QA`.

        Notes:
            If the predicted query cannot be run on the db, the resulting metrics are all zeros

        Examples:
            >>> test = {'db_id': 'database1', 'target': 'SELECT DISTINCT emailisfree FROM fraud', 'prediction': 'SELECT emailsisfree, income FROM fraud'}
            >>> evaluator = MetricEvaluator(databases)
            >>> results = evaluator.evaluate_single_test_SP(test, 'prediction', 'target')
            >>> print(results)
            {'cell_precision_prediction': 0.50, 'cell_recall_prediction': 1.0}
        """
        # Compares the target and predicted SQL queries after cleaning and formatting. If they are identical, it returns metrics as 1
        if self.are_cleaned_sql_identical(test[target_col_name], test[prediction_col_name]):
            metrics_result = {f'{metric}_{prediction_col_name}': 1 for metric in self.metrics}
            metrics_result[f'tuple_order_{prediction_col_name}'] = None \
                if 'order' not in test[target_col_name].lower() else 1
            return metrics_result

        # Tries to run the predicted query on the database. If there is an error (e.g. syntax error in the query),
        # it logs the error and returns metrics as 0
        try:
            test[prediction_col_name] = self.databases.run_query(test['db_id'], test[prediction_col_name])
        except sqlite3.Error as e:
            # catch any possible error of prediction and return all zeros
            logging.error(e)
            return {f'{metric}_{prediction_col_name}': 0 for metric in self.metrics}
        # Evaluates the results of the target and predicted queries using the evaluate_single_test_QA function
        return self.evaluate_single_test_QA(test, prediction_col_name, target_col_name)

    @staticmethod
    def are_cleaned_sql_identical(target: str, prediction: str) -> bool:
        """
        Create a mask based on whether the target and prediction strings are equal after cleaning.

        Args:
            target (str): The target string.
            prediction (str): The prediction string.

        Returns:
            bool: True if cleaned prediction equals cleaned target, False otherwise.
        """
        new_target = (target.lower()
                      .replace(" ,", ",")
                      .replace("  ", " ")
                      .replace('"', '')
                      .replace("'", '')
                      .strip())

        new_pred = (prediction.lower()
                    .replace(" ,", ",")
                    .replace("  ", " ")
                    .replace('"', '')
                    .replace("'", '')
                    .replace(' ( ', '(')
                    .replace(' )', ')')
                    .strip())
        return new_pred == new_target
