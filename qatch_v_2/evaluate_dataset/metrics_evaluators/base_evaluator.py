from __future__ import annotations

import math
import random
import re
from abc import ABC, abstractmethod

import numpy as np
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

        Args:
            cell: The cell value to normalize.

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
