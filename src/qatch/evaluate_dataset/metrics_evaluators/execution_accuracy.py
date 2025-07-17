from __future__ import annotations

from .base_evaluator import BaseEvaluator
from .utils import sort_with_different_types


class ExecutionAccuracy(BaseEvaluator):
    @property
    def metric_name(self):
        return 'execution_accuracy'

    def run_metric(self, target: list[list], prediction: list[list], *args, **kwargs) -> float | int:
        """
        Calculates the execution accuracy between the target and prediction.
        The logic comes from: "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain
        Semantic Parsing and Text-to-SQL Task"

        This function compares a set of target rows with the prediction by creating a unique representation of data sequence.
        If the matching rows are not found, it returns 0.0 else it continues to check other rows.
        If all rows match perfectly returns 1.0, else it returns 0.0.

        Args:
            *args:
            **kwargs:
            target (list[list]): The target list of lists, each representing a unique sequence of data.
            prediction (list[list]): The prediction list of lists, each representing a matched sequence of data.

        Returns:
            float | int: Returns 1.0 if all rows match perfectly, otherwise returns 0.0.

        Note:
            - The function uses the function sort_with_different_types to sort data having different data types.
            - The metric is robust to different tuple order between predictions and target
            - The metric is robust to different projection order (Name, Surname) = (Surname, Name)
        """
        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len == 0:
            return 1.0

        if len(target) != len(prediction):
            return 0.0

        gold_row_set = set()
        pred_row_set = set()

        for gold_row, predicted_row in zip(target, prediction):
            gold_row_set.add(tuple(sort_with_different_types(gold_row)))
            pred_row_set.add(tuple(sort_with_different_types(predicted_row)))

        return int(gold_row_set == pred_row_set)
