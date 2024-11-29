from __future__ import annotations

import random
from abc import ABC, abstractmethod

from typing_extensions import Literal, TypedDict

from ..state_orchestrator_evaluator import StateOrchestratorEvaluator


class EvaluatedTest(TypedDict):
    metric_value: float | int
    metric_name: str


class BaseEvaluator(ABC):
    def __init__(self, seed=2023):
        random.seed(seed)
        self.connector = None

    @property
    @abstractmethod
    def metric_name(self):
        raise NotImplementedError

    def graph_call(self, state: StateOrchestratorEvaluator) -> dict[Literal['evaluated_tests']: EvaluatedTest]:
        predicted_test = state['predicted_test']
        evaluated_test = self._wrapper_run_metric(target=predicted_test.target, prediction=predicted_test.prediction)
        return {'evaluated_tests': [EvaluatedTest(metric_name=self.metric_name, metric_value=evaluated_test)]}

    @abstractmethod
    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        raise NotImplementedError

    def _wrapper_run_metric(self, target: list[list], prediction: list[list]) -> float:
        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len:
            return 1.0
        if target_len == 0 and prediction_len != 0:
            return 0.0
        if prediction_len == 0 and target_len != 0:
            return 0.0
        else:
            return self.run_metric(target=target, prediction=prediction)
