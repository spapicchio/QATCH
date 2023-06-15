import itertools

import numpy as np

from .abstract_metric import AbstractMetric


class CellPrecisionTag(AbstractMetric):
    def evaluate_single_no_special_case(self, target: list[list],
                                        prediction: list[list]) -> float:
        """
        the ratio of predicted cell that are in the target.
        Does not consider cardinality (measured by other tags)
        • High PRECISION means that the model is good at
          identifying relevant instances and has a low false positive rate.

        :param target: target table to be compared with prediction table
        :param prediction: prediction table to be compared with target table
        :return: score between [0,1]
            * 0 indicates no cell in the prediction is in the target
            * 1 indicates all cells in the prediction are in the target
        """
        flat_target = list(itertools.chain(*target))
        flat_prediction = list(itertools.chain(*prediction))

        sum_cell_match = np.sum(np.isin(flat_prediction, flat_target))

        return round(sum_cell_match / len(flat_prediction), 3)
