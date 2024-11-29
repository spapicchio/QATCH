from __future__ import annotations

from collections import Counter

from .base_evaluator import BaseEvaluator


class TupleConstraint(BaseEvaluator):
    @property
    def metric_name(self):
        return 'tuple_constraint'

    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        target = map(sorted, target)
        prediction = map(sorted, prediction)

        target = map(tuple, target)
        prediction = map(tuple, prediction)

        count_targ_dict = Counter(target)
        count_pred_dict = Counter(prediction)

        cardinality = [count_pred_dict[key] == count for key, count in count_targ_dict.items()]

        return round(sum(cardinality) / len(cardinality), 3)
