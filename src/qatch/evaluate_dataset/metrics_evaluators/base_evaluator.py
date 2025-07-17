from __future__ import annotations

import random
from abc import ABC, abstractmethod

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
        self.connector: BaseConnector | None = None

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
        evaluated_test = self.run_metric(
            target=predicted_test.target_values,
            prediction=predicted_test.predicted_values,
            target_query=predicted_test.target_query,
            predicted_query=predicted_test.predicted_query,
            connector=state['connector']
        )
        return {'evaluated_tests': [EvaluatedTest(metric_name=self.metric_name, metric_value=evaluated_test)]}

    @abstractmethod
    def run_metric(self, target: list[list], prediction: list[list], *args, **kwargs) -> float | int:
        """
        Abstract method to compute a specific evaluation metric based on targets and predictions.

        Args:
            target (list[list]): The ground truth values to be compared against the predictions.
                                 Each sublist represents a different set of associated target values.

            prediction (list[list]): The predicted values to be evaluated against the target.
                                     Each sublist represents a different set of associated predicted values.

            *args (Any): Additional arguments to be passed to the evaluation method.:
            **kwargs (Any): Additional keyword arguments to be passed to the evaluation method.:

        Returns:
            float | int: The computed evaluation metric value. The actual type and semantics of the result
                         (whether it's a score, an error, etc.) depend on the specific implementation in a subclass.

        Note:
            - This is an abstract method and needs to be implemented in any concrete subclass of `BaseEvaluator`.
            - it's intended to be invoked indirectly via the `_wrapper_run_metric` method, which takes care
            of normalizing inputs before passing them to this method.
        """

        raise NotImplementedError
