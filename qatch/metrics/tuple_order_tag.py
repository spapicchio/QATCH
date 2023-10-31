import numpy as np

from .abstract_metric import AbstractMetric


class TupleOrderTag(AbstractMetric):
    def evaluate_single_no_special_case(self,
                                        target: list[list],
                                        prediction: list[list]) -> float:
        """
        The score is based on the spearman rank correlation coefficient.
        see: https://stackoverflow.com/questions/859536/sorted-list-difference/859902#859902
        • If some elements are duplicated, consider only one
        • If some elements are not present, they are not considered
            target = [a, a, a, b, c, d], pred = [b, b, a, c, e]
            new_targ = [a, b, c], pred = [b, a, c]

        Example:
            >> target = [['apple', 'orange'], ['pear']]
            >> prediction = [['pear'], ['apple', 'orange']]
            >> -1
        :param target: target table to be compared with prediction table
        :param prediction: prediction table to be compared with target table
        :return: score between [-1, 1]
            * 0 indicates that there is no correlation between the two lists
            * 1 indicates that the order of rows in prediction is same as target
            * -1 indicates order of rows in prediction is opposite to target
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
