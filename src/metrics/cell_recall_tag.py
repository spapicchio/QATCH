import itertools

import numpy as np

from .abstract_metric import AbstractMetric


def check_isin(cell, target):
    return cell in target


class CellRecallTag(AbstractMetric):
    def evaluate_single_no_special_case(self, target: list[list],
                                        prediction: list[list]) -> float | str:
        """
        the ratio of target cell that are in the prediction.
        • High RECALL means that the model is good at
          identifying all relevant instances and has a low false negative rate.

        :param target: target table to be compared with prediction table
        :param prediction: prediction table to be compared with target table
        :return: score between [0,1]
            * 0 indicates no cell in the target is in the prediction
            * 1 indicates all cells in the target are in the prediction
        """
        flat_target = set(itertools.chain(*target))
        flat_prediction = set(itertools.chain(*prediction))

        # sum_cell_match = np.sum(np.isin(flat_target, flat_prediction))
        sum_cell_match = np.sum(list(map(
            lambda x: check_isin(x, flat_prediction),
            flat_target
        )))

        return round(sum_cell_match / len(flat_target), 3)
