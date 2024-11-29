from __future__ import annotations

from .base_evaluator import BaseEvaluator


class ExecutionAccuracy(BaseEvaluator):
    @property
    def metric_name(self):
        return 'execution_accuracy'

    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        if len(target) != len(prediction):
            return 0.0

        for gold_row, predicted_row in zip(target, prediction):
            if tuple(sort_with_different_types(gold_row)) != tuple(sort_with_different_types(predicted_row)):
                return 0.0

        return 1.0


def sort_key(x):
    if x is None:
        return 0, ''  # Treat None as the smallest value
    elif isinstance(x, (int, float)):
        return 1, float(x)  # Handle numerical types uniformly
    else:
        return 2, str(x)  # Convert all other types to string for consistent comparison


def sort_with_different_types(arr):
    sorted_arr = sorted(arr, key=sort_key)
    return sorted_arr
