from __future__ import annotations

import numpy as np

from .base_evaluator import BaseEvaluator


class TupleOrder(BaseEvaluator):
    @property
    def metric_name(self):
        return 'tuple_order'

    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        # take only prediction that are in target without duplicates
        # MAINTAINING the order
        new_pred = []
        [new_pred.append(pred) for pred in prediction
         if pred in target and pred not in new_pred]
        # same for target
        new_target = []
        [new_target.append(tar) for tar in target
         if tar in prediction and tar not in new_target]

        if len(new_target) == 0:

            rho = 0.0
        else:
            target_ranks = [i for i in range(len(new_target))]
            pred_ranks = [new_target.index(row) for row in new_pred]

            diff_rank_squared = [(tar - pred) ** 2
                                 for tar, pred in zip(target_ranks, pred_ranks)]

            sum_diff_rank_squared = sum(diff_rank_squared)

            n = len(new_target) if len(new_target) > 1 else 2
            rho = 1 - 6 * sum_diff_rank_squared / (n * (n ** 2 - 1))

        return self.normalize(round(rho, 3))

    @staticmethod
    def normalize(data: float):
        data = [-1, data, 1]
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data[1]
