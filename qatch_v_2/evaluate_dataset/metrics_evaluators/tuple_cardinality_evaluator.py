from __future__ import annotations

from .base_evaluator import BaseEvaluator


class TupleCardinality(BaseEvaluator):
    @property
    def metric_name(self):
        return 'tuple_cardinality'

    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        if len(prediction) >= len(target):
            # in case we have more elements in the prediction than in the target
            return round(len(target) / len(prediction), 3)

        # in case we have more elements in the target than in the prediction
        return round(len(prediction) / len(target), 3)
