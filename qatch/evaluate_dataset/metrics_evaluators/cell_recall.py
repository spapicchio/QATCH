from __future__ import annotations

from itertools import chain

from .base_evaluator import BaseEvaluator


class CellRecall(BaseEvaluator):
    @property
    def metric_name(self):
        return 'cell_recall'

    def run_metric(self, target: list[list], prediction: list[list], *args, **kwargs) -> float | int:
        """
        Calculates the ratio of target cells that are in the prediction.
        High recall indicates that the model is good at identifying all relevant instances
        and has a low false negative rate.

        Args:
            *args:
            **kwargs:
            target (list[list]): Target table to be compared with the prediction table.
            prediction (list[list]): Prediction table to be compared with the target table.

        Returns:
            float: Recall score between [0, 1].
                - 0 indicates no cell in the target is in the prediction.
                - 1 indicates all cells in the target are in the prediction.

        Examples:
            >>> evaluator = CellRecall()
            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['c', 'd']
            >>> evaluator.run_metric(target,prediction)
            1.0

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'x'], ['y', 'd']]
            >>> evaluator.run_metric(target,prediction)
            0.5

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'a'], ['b', 'b'], ['c', 'd']]
            >>> evaluator.run_metric(target,prediction)
            1.0
        """
        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len == 0:
            return 1.0
        if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
            return 0.0

        target = set(chain.from_iterable(target))
        prediction = set(chain.from_iterable(prediction))
        intersected_cells = target.intersection(prediction)
        sum_cell_match = len(intersected_cells)
        return round(sum_cell_match / len(target), 3)
