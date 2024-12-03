from __future__ import annotations

import numpy as np

from .base_evaluator import BaseEvaluator


class TupleOrder(BaseEvaluator):
    @property
    def metric_name(self):
        return 'tuple_order'

    def run_metric(self, target: list[list], prediction: list[list], *args, **kwargs) -> float | int:
        """
        Evaluates the similarity in tuple order between the target and prediction.
        The score is based on the Spearman rank correlation coefficient normalized between 0 and 1.
        This metric ONLY checks whether the order of the tuples is the same in the target and prediction.
        Therefore, the elements that are in predictions but nor in target are ignored (and viceversa).

        Args:
            *args:
            **kwargs:
            target (list[list]): Target table to be compared with the prediction table.
            prediction (list[list]): Prediction table to be compared with the target table.

        Returns:
            float: Score between [-1, 1].
            - 1 indicates that the order of rows in prediction is the same as in the target.
            - 0.5 indicates that there is no correlation between the two lists.
            - 0 indicates the order of rows in prediction is opposite to the target.

        Examples:
            >>> evaluator = TupleOrder()
            >>>  target = [['a', 'b'], ['c', 'd']]
            >>>  prediction = [['c', 'd'], ['a', 'b']]
            >>> evaluator.run_metric(target,prediction)
            0.0

            >>>  target = [['apple', 'orange'], ['pear']]
            >>>  prediction = [['pear'], ['apple', 'orange']]
            >>> evaluator.run_metric(target,prediction)
            0.0

            >>>  target = [['apple', 'orange'], ['pear']]
            >>>  prediction = [['pear']]
            >>> evaluator.run_metric(target,prediction)
            1.0
        """
        target_len = len(target)
        prediction_len = len(prediction)
        if target_len == prediction_len == 0:
            return 1.0
        if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
            return 0.0

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
            # case when prediction does not have any element in target
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
