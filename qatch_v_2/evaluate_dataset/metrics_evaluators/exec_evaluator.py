from __future__ import annotations

from .base_evaluator import BaseEvaluator


class ExecutionAccuracy(BaseEvaluator):
    @property
    def metric_name(self):
        return 'execution_accuracy'

    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        if len(target) != len(prediction):
            return 0.0

        gold_row_set = set()
        pred_row_set = set()

        for gold_row, predicted_row in zip(target, prediction):
            gold_row_set.add(tuple(sort_with_different_types(gold_row)))
            pred_row_set.add(tuple(sort_with_different_types(predicted_row)))

        while len(pred_row_set) > 0:
            pred_row = pred_row_set.pop()
            if pred_row in gold_row_set:
                gold_row_set.remove(pred_row)
            else:
                return 0.0

        return 1.0 if len(gold_row_set) == 0 else 0.0


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
