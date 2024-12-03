from __future__ import annotations

from itertools import chain

from .base_evaluator import BaseEvaluator


class CellPrecision(BaseEvaluator):
    @property
    def metric_name(self):
        return 'cell_precision'

    def run_metric(self, target: list[list], prediction: list[list], *args, **kwargs) -> float | int:
        """
        Calculates the ratio of predicted cells that are in the target.
        Does not consider cardinality (measured by other tags).
        High precision indicates that the model is good at identifying relevant instances
        and has a low false positive rate.

        Args:
            *args:
            **kwargs:
            target (list[list]): Target table to be compared with the prediction table.
            prediction (list[list]): Prediction table to be compared with the target table.

        Returns:
            float: Precision score between [0, 1].
                - 0 indicates no cell in the prediction is in the target.
                - 1 indicates all cells in the prediction are in the target.

        Examples:
            >>> evaluator = CellPrecision()
            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['c', 'd']
            >>> evaluator.run_metric(target,prediction)
            1.0

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['c', 'e']
            >>> evaluator.run_metric(target,prediction)
            0.75

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a'], ['b'], ['c'], ['d']]
            >>> evaluator.run_metric(target,prediction)
            1.0  # it is one even if the schema does not match (we introduce tuple constraints for this)
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
        return round(sum_cell_match / len(prediction), 3)
