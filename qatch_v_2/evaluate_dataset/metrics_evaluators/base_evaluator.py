from __future__ import annotations

import math
import random
import re
from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Literal, TypedDict

from ..state_orchestrator_evaluator import StateOrchestratorEvaluator
from ...connectors import BaseConnector


class EvaluatedTest(TypedDict):
    metric_value: float | int
    metric_name: str


class BaseEvaluator(ABC):
    """
    A Base class for all types of evaluators. This class provides the basic skeleton for a metric
    evaluator. It includes a method to evaluate a test using a graph call (`graph_call`), and an
    abstract method (`run_metric`) that must be implemented by each child class for specific metric
    calculation. It also includes a method to handle edge cases and preprocess inputs
    (`_wrapper_run_metric`), and a static method to normalize cell values (`normalize_cell`).

    Note:
        This class should not be instantiated directly. Instead, use it as a base class for all
        evaluator classes. Each child class must implement the `run_metric` method.

    Attributes:
        seed (int): A seed value used for any randomness in the evaluator.
        connector: A context holder for evalutors. Sub-classes can use this to hold any required state/data.
    """

    def __init__(self, seed=2023):
        random.seed(seed)
        self.connector: BaseConnector = None

    @property
    @abstractmethod
    def metric_name(self):
        """This name represent the metric_name in the output"""
        raise NotImplementedError

    def graph_call(self, state: StateOrchestratorEvaluator) -> dict[Literal['evaluated_tests']: EvaluatedTest]:
        """
        Executes a test evaluation in the context of the given state and returns the evaluated test encapsulated
        in a dictionary.

        The method uses the 'predicted_test' value from the provided state to calculate the metric value.
        This metric value along with the metric name are encapsulated into an `EvaluatedTest`
        object which is returned in a dictionary.

        Args:
            state (StateOrchestratorEvaluator): The state in which the test is executed. It must contain the 'predicted_test'
                                                information.

        Returns:
            dict[Literal['evaluated_tests']: EvaluatedTest]: A dictionary with the key 'evaluated_tests' and the value being a
                                                             list containing a single `EvaluatedTest` object.

        Note:
            - This method is the one used in the LangGraph graph in `OrchestratorEvaluator`
            - This method relies on the `_wrapper_run_metric` for calculating the metric value. Make sure the
            `predicted_test` information in the state is accurate and compatible with the evaluation metric used in
             `_wrapper_run_metric`.
        """

        predicted_test = state['predicted_test']
        evaluated_test = self._wrapper_run_metric(target=predicted_test.target, prediction=predicted_test.prediction)
        return {'evaluated_tests': [EvaluatedTest(metric_name=self.metric_name, metric_value=evaluated_test)]}

    @abstractmethod
    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        """
        Abstract method to compute a specific evaluation metric based on targets and predictions.

        Args:
            target (list[list]): The ground truth values to be compared against the predictions.
                                 Each sublist represents a different set of associated target values.

            prediction (list[list]): The predicted values to be evaluated against the target.
                                     Each sublist represents a different set of associated predicted values.

        Returns:
            float | int: The computed evaluation metric value. The actual type and semantics of the result
                         (whether it's a score, an error, etc.) depend on the specific implementation in a subclass.

        Note:
            - This is an abstract method and needs to be implemented in any concrete subclass of `BaseEvaluator`.
            - it's intended to be invoked indirectly via the `_wrapper_run_metric` method, which takes care
            of normalizing inputs before passing them to this method.
        """

        raise NotImplementedError

    def _wrapper_run_metric(self, target: list[list], prediction: list[list]) -> float:
        """
        Wraps the `run_metric` method by providing additional handling for edge cases and pre-processing steps.

        Args:
            target (list[list]): The target values as a nested list which is to be compared against predictions.
                                 Each child list represents a different set of associated target values.

            prediction (list[list]): The prediction values as a nested list which are to be evaluated against targets.
                                     Each child list represents a different set of associated predicted values.

        Returns:
            float: The evaluation metric computed by the `run_metric` method. Returns 1.0 if both target and prediction
                   are empty, 0.0 in case only one them is empty.

        Note:
            Both `target` and `prediction` inputs are normalized before passing them to the `run_metric` method.
            The normalization process depends on the specific implementation of the `normalize_cell` static method.

            This is an internal function and intended to be used only within the class. It serves as a
            preparation stage for the `run_metric` method.
        """

        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len == 0:
            return 1.0
        if target_len == 0 and prediction_len != 0:
            return 0.0
        if prediction_len == 0 and target_len != 0:
            return 0.0
        else:
            # normalize target and prediction
            target = [list(map(self.normalize_cell, row)) for row in target]
            prediction = [list(map(self.normalize_cell, row)) for row in prediction]
            return self.run_metric(target=target, prediction=prediction)

    @staticmethod
    def normalize_cell(cell):
        """
        Normalizes a cell value for comparison.

        This method takes in a cell value and normalizes it based on its type, such as boolean, float,
        int and string. It handles different behaviors if the cell value is None or if it is a number
        represented as a string. It performs rounding of numbers to a precision of two decimal places.
        In case of strings, they are stripped of leading and trailing whitespaces and converted to
        lower case.

        Note:
        The normalized cell value is always returned in string format, even for boolean values
        and None.

        Args:
            cell (bool, float, int, str or None): The cell value to normalize. It could be a boolean,
        float, int, str or None.

        Returns:
            str: The normalized cell value.
        """
        if cell is None:
            return "None"
        elif isinstance(cell, bool):
            return cell
        elif not isinstance(cell, str) and math.isnan(cell):
            return 'None'
        elif isinstance(cell, (np.float_, np.int_)):
            cell = str(round(cell, 2))
        elif isinstance(cell, (float, int)):
            cell = str(round(cell, 2))
        elif isinstance(cell, str) and re.match(r'^-?\d+(?:\.\d+)?$', cell):
            # number as string
            # round only if it has decimal places
            if '.' in cell:
                cell = str(round(float(cell), 2))
            else:
                cell = str(int(cell))
        else:
            # string with no numbers
            cell = cell.replace('\n', '').strip().lower()
        return cell
