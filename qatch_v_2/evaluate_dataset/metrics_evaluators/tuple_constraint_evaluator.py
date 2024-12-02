from __future__ import annotations

from collections import Counter

from .base_evaluator import BaseEvaluator


class TupleConstraint(BaseEvaluator):
    @property
    def metric_name(self):
        return 'tuple_constraint'

    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        """
        Evaluates the ratio between the cardinality of the target tuples and the prediction.
        Returns a score between 0 and 1. It is 1 if the schema, the cardinality and the cell values are equal.

        Args:
            target (list[list]): Target table to be compared with the prediction table.
            prediction (list[list]): Prediction table to be compared with the target table.

        Returns:
            float: Score between [0, 1].
                - 0 indicates NONE of the schema/cardinality/cell_values  are the same in prediction.
                - 1 indicates the schema, the cardinality and the cell values of
                    the prediction tuples are equal to the target ones.

        Examples:
            >>> evaluator = TupleConstraint()
            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['c', 'd']]
            >>> evaluator.run_metric(target, prediction)
            1.0

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['a', 'b'], ['c', 'd']]
            >>> evaluator.run_metric(target, prediction)
            0.5  # only ['c', 'd'] is the same in both tables

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['a', 'b'], ['c', 'd'], ['c', 'd']]
            >>> evaluator.run_metric(target, prediction)
            0.0
        """

        target = map(sorted, target)
        prediction = map(sorted, prediction)

        target = map(tuple, target)
        prediction = map(tuple, prediction)

        count_targ_dict = Counter(target)
        count_pred_dict = Counter(prediction)

        cardinality = [count_pred_dict[key] == count for key, count in count_targ_dict.items()]

        return round(sum(cardinality) / len(cardinality), 3)
