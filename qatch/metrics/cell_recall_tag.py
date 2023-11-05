import numpy as np

from .abstract_metric import AbstractMetric


def check_isin(cell, target):
    return cell in target


class CellRecallTag(AbstractMetric):
    def evaluate_single_no_special_case(self, target: list[list],
                                        prediction: list[list]) -> float:
        """
        Calculates the ratio of target cells that are in the prediction.
        High recall indicates that the model is good at identifying all relevant instances
        and has a low false negative rate.

        Args:
            target (list[list]): Target table to be compared with the prediction table.
            prediction (list[list]): Prediction table to be compared with the target table.

        Returns:
            float: Recall score between [0, 1].
                - 0 indicates no cell in the target is in the prediction.
                - 1 indicates all cells in the target are in the prediction.

        Examples:
            >>> evaluator = CellRecallTag()
            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['c', 'd']
            >>> evaluator.evaluate_single_no_special_case(target, prediction)
            1.0

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'x'], ['y', 'd']]
            >>> evaluator.evaluate_single_no_special_case(target, prediction)
            0.5

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'a'], ['b', 'b'], ['c', 'd']]
            >>> evaluator.evaluate_single_no_special_case(target, prediction)
            1.0
        """
        target = np.array(target)
        prediction = np.array(prediction)
        sum_cell_match = np.sum(np.isin(target, prediction))
        return round(sum_cell_match / target.size, 3)
