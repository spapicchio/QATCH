from __future__ import annotations

from itertools import chain

from .base_evaluator import BaseEvaluator


class CellRecall(BaseEvaluator):
    @property
    def metric_name(self):
        return 'cell_recall'

    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        target = set(chain.from_iterable(target))
        prediction = set(chain.from_iterable(prediction))
        intersected_cells = target.intersection(prediction)
        sum_cell_match = len(intersected_cells)
        return round(sum_cell_match / len(target), 3)
