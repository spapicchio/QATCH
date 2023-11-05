import numpy as np

from .abstract_metric import AbstractMetric


class TupleOrderTag(AbstractMetric):
    def evaluate_single_no_special_case(self,
                                        target: list[list],
                                        prediction: list[list]) -> float:
        """
        Evaluates the similarity in tuple order between the target and prediction.
        The score is based on the Spearman rank correlation coefficient normalized between 0 and 1.
        This metric ONLY checks whether the order of the tuples is the same in the target and prediction.
        Therefore, the elements that are in predictions but nor in target are ignored (and viceversa).

        Args:
            target (list[list]): Target table to be compared with the prediction table.
            prediction (list[list]): Prediction table to be compared with the target table.

        Returns:
            float: Score between [-1, 1].
            - 1 indicates that the order of rows in prediction is the same as in the target.
            - 0.5 indicates that there is no correlation between the two lists.
            - 0 indicates the order of rows in prediction is opposite to the target.

        Examples:
            >>> evaluator = TupleOrderTag()
            >>>  target = [['a', 'b'], ['c', 'd']]
            >>>  prediction = [['c', 'd'], ['a', 'b']]
            >>> evaluator.evaluate(target, prediction)
            0.0

            >>> evaluator = TupleOrderTag()
            >>>  target = [['apple', 'orange'], ['pear']]
            >>>  prediction = [['pear'], ['apple', 'orange']]
            >>> evaluator.evaluate(target, prediction)
            0.0

            >>> evaluator = TupleOrderTag()
            >>>  target = [['apple', 'orange'], ['pear']]
            >>>  prediction = [['pear']]
            >>> evaluator.evaluate(target, prediction)
            1.0
        """

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
