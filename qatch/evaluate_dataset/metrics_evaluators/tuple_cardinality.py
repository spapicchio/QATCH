from __future__ import annotations

from .base_evaluator import BaseEvaluator


class TupleCardinality(BaseEvaluator):
    @property
    def metric_name(self):
        return 'tuple_cardinality'

    def run_metric(self, target: list[list], prediction: list[list], *args, **kwargs) -> float | int:
        """
        Evaluates the ratio of the length of the smaller list to the length of the larger list.

        Calculates the ratio of the length of the target table to the length of the prediction table
        or vice-versa based on the maximum length to ensure the score falls between 0 and 1.

        Args:
            target (list[list]): Target table to be compared with the prediction table.
            prediction (list[list]): Prediction table to be compared with the target table.

        Returns:
            float: Score between [0, 1].
                - 0 indicates the target/prediction is zero and the other is not.
                - 1 indicates the target/prediction is the same size as the other.

        Examples:
            >>> evaluator = TupleCardinality()
            >>> target = [['a', 'b'], ['c', 'd'], ['c', 'd'], ['f', 'g']]
            >>> prediction = [['a', 'b'], [3, 2]]
            >>> evaluator.run_metric(target,prediction)
            0.5  # 2/4

            >>> target = [['a', 'b'], [3, 2]]
            >>> prediction = [['a', 'b'], ['c', 'd'], ['c', 'd'], ['f', 'g']]
            >>> evaluator.run_metric(target,prediction)
            0.5

            >>> target = [['a', 'b'], [3, 2]]
            >>> prediction = [['a', 'b'], ['c', 'd']]
            >>> evaluator.run_metric(target,prediction)
            1.0
        """
        if len(target) == len(prediction) == 0:
            return 1.0

        if len(prediction) >= len(target):
            # in case we have more elements in the prediction than in the target
            return round(len(target) / len(prediction), 3)

        # in case we have more elements in the target than in the prediction
        return round(len(prediction) / len(target), 3)
