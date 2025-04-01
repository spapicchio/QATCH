from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd
from func_timeout import FunctionTimedOut
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from sqlalchemy.exc import CompileError, DBAPIError, OperationalError
from tqdm import tqdm

from .metrics_evaluators import (
    CellPrecision,
    CellRecall,
    TupleCardinality,
    TupleOrder,
    TupleConstraint,
    ExecutionAccuracy,
    ValidEfficiencyScore,
)
from .state_orchestrator_evaluator import StateOrchestratorEvaluator, PredictedTest
from ..connectors import BaseConnector, SqliteConnector

name2evaluator = {
    "cell_precision": CellPrecision,
    "cell_recall": CellRecall,
    "tuple_cardinality": TupleCardinality,
    "tuple_order": TupleOrder,
    "tuple_constraint": TupleConstraint,
    "execution_accuracy": ExecutionAccuracy,
    "VES": ValidEfficiencyScore,
}


def _utils_run_query_if_str(
    query: str | list[list], connector: BaseConnector
) -> list[list] | None:
    """
    This method takes a SQL query or a list of lists and an instance of a BaseConnector.
    If the query is a string - it attempts to run the query on a supplied connector, handling SQL exceptions and edge cases.
    If the query is a list - the function simply returns the query.

    Args:
        query (str|list[list]): Input query or data in the form of string or list of lists.
        connector (BaseConnector): A BaseConnector object to execute SQL queries on.

    Returns:
        list[list] | None: Executed query results as a list of lists if query string has been passed.
                            Passed query if a list of lists has been passed.
                            None if the query execution resulted in a SQL exception.

    Note:
        String type queries will have ';' removed before execution for safety reasons.

        This function handles SQL exceptions like CompileError, DBAPIError, FunctionTimedOut and OperationalError.
        In case of an exception, the function will log a warning and return None.
    """

    if not isinstance(query, str):
        return query

    query = query.replace(";", "")
    try:
        result = connector.run_query(query)
        return result
    except (CompileError, DBAPIError, FunctionTimedOut, OperationalError) as e:
        logging.warning(e)


class OrchestratorEvaluator:
    """
    A class that evaluates metrics on test cases using different evaluators.

    This class preprocesses test cases, organizes them in a LangGraph object to execute in parallel the selected metrics

    Note:
        - The class can accept a predefined list of evaluator names. If no names are provided,
          it uses all available evaluators.
        - The evaluation proceeds in parallel for speeding up the execution.

    Attributes:
        evaluator_names (list[str] | None): A list of evaluator names to use for evaluation, or None to use all.
        graph (StateGraph[StateOrchestratorEvaluator]): The internal state graph holding the evaluators.

    """

    def __init__(self, evaluator_names: list[str] | None = None):
        graph = StateGraph(StateOrchestratorEvaluator)
        self.evaluator_names = evaluator_names or list(name2evaluator.keys())
        list_node_fun = [
            (name, name2evaluator[name]().graph_call) for name in self.evaluator_names
        ]

        for node_name, node_fun in list_node_fun:
            graph.add_node(node_name, node_fun)
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        self.graph = graph.compile()

    def evaluate_df(
        self,
        df: pd.DataFrame,
        target_col_name: str,
        prediction_col_name: str,
        db_path_name: str,
    ) -> pd.DataFrame:
        """
        Evaluates a dataframe with test predictions applying suitable metrics. The function transforms the input
        dataframe into a dictionary and processes each test case individually. It connects to SQL database and
        computes metrics for each test case. The computed metrics are added to the dataframe that gets returned.

        Args:
            df (pd.DataFrame): The input dataframe containing the test predictions.
            target_col_name (str): The name of the column in the dataframe that contains the target results.
            prediction_col_name (str): The name of the column in the dataframe that contains the predicted results.
            db_path_name (str): The name of the field in the dataframe that holds the database path.

        Returns:
            pd.DataFrame : The input dataframe enriched with the metrics computed for each test case.

        Note:
            - To speedup execution, the evaluation is performed for each database sequentially
            in order to not recreate connection.
        """

        df_dict = df.to_dict("records")

        # create dictionary of db_path to tests. This is used to spedup execution
        db_path2tests = defaultdict(list)
        for test in df_dict:
            db_path2tests[test[db_path_name]].append(test)

        for db_path, tests in db_path2tests.items():
            # create a connection only once for each test
            connector = SqliteConnector(relative_db_path=db_path, db_name="_")

            for test in tqdm(
                tests, desc=f"Evaluating tests for {db_path.split('/')[-1]}"
            ):
                metrics = self.evaluate_single_test(
                    test[target_col_name], test[prediction_col_name], connector
                )

                for metric, value in metrics.items():
                    test[metric] = value

        return pd.DataFrame(df_dict)

    def evaluate_single_test(
        self,
        target_query: str | list[list],
        predicted_query: str | list[list],
        connector: BaseConnector,
    ) -> dict:
        """
        Evaluates a single test/query pair by comparing the predicted results to the expected target.

        The method first checks if the input queries are strings (SQL code) and if the target query contains
        an 'order by' clause.

        The comparison of the target query with the predicted one is done in a case-insensitive manner.
        Therefore, if both queries are strings and they are equal, the metrics value is set to 1.0.

        If the input queries are strings, then these queries are run via a database connector.
        The results of the queries are then passed to `PredictedTest` and `self.graph.invoke`.

        Note:
            - Tuple Order is computed only if target SQL contains an 'oder by' clause.
            - Some metrics can be computed only for Text2SQL as VES.

        Args:
            target_query (str | list[list]): The target SQL query or the expected results in a nested list.
            predicted_query (str | list[list]): The predicted SQL query or the predicted results in a nested list.
            connector (BaseConnector): An object of BaseConnector class to communicate with the database.

        Returns:
            dict: A dictionary comprising the evaluation metrics values for the test.

        Raises:
            ValueError: If an error is encountered while running the target query.
        """

        # Check if queries are strings and, if so, whether the target query contains an order by clause
        is_order = isinstance(target_query, str) and "order by" in target_query.lower()

        # Assume metrics2value to be 0.0 unless proven otherwise
        metrics2value = {name: 0.0 for name in self.evaluator_names}

        # Check if both queries are strings and equal.
        if (
            isinstance(target_query, str)
            and isinstance(predicted_query, str)
            and target_query.lower() == predicted_query.lower()
        ):
            metrics2value = {name: 1.0 for name in self.evaluator_names}
        else:
            # Run queries if they're strings
            target_values = _utils_run_query_if_str(target_query, connector)
            predicted_values = _utils_run_query_if_str(predicted_query, connector)

            if target_values is None:
                raise ValueError(f"Target gets an Error `{target_query}`")

            if isinstance(predicted_query, list):
                predicted_query = ""

            if isinstance(target_query, list):
                target_query = ""

            if predicted_values is not None:
                predicted_test = PredictedTest(
                    target_query=target_query,
                    target_values=target_values,
                    predicted_query=predicted_query,
                    predicted_values=predicted_values,
                )
                state = self.graph.invoke(
                    {"predicted_test": predicted_test, "connector": connector}
                )
                metrics2value = self._parse_graph_output(state)

        metrics2value["tuple_order"] = (
            metrics2value["tuple_order"] if is_order else None
        )

        return metrics2value

    def _parse_graph_output(self, state: StateOrchestratorEvaluator) -> dict:
        """parse function that connects the Graph State with the columns to add in a pd.DataFrame"""
        evaluated_tests = state["evaluated_tests"]
        output = dict()
        for test in evaluated_tests:
            output[test["metric_name"]] = test["metric_value"]
        return output

    def _is_target_equal_to_pred(self, target: str, prediction: str):
        """Check if target is equal to prediction. In future release,
        this will be substitute with more sophisticated syntactic metrics"""
        return target.lower() == prediction.lower()