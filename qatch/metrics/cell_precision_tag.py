import numpy as np

from .abstract_metric import AbstractMetric


class CellPrecisionTag(AbstractMetric):
    def evaluate_single_no_special_case(self, target: list[list],
                                        prediction: list[list]) -> float:
        """
        Calculates the ratio of predicted cells that are in the target.
        Does not consider cardinality (measured by other tags).
        High precision indicates that the model is good at identifying relevant instances
        and has a low false positive rate.

        Args:
            target (list[list]): Target table to be compared with the prediction table.
            prediction (list[list]): Prediction table to be compared with the target table.

        Returns:
            float: Precision score between [0, 1].
                - 0 indicates no cell in the prediction is in the target.
                - 1 indicates all cells in the prediction are in the target.

        Examples:
            >>> evaluator = CellPrecisionTag()
            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['c', 'd']
            >>> evaluator.evaluate_single_no_special_case(target, prediction)
            1.0

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a', 'b'], ['c', 'e']
            >>> evaluator.evaluate_single_no_special_case(target, prediction)
            0.75

            >>> target = [['a', 'b'], ['c', 'd']]
            >>> prediction = [['a'], ['b'], ['c'], ['d']]
            >>> evaluator.evaluate_single_no_special_case(target, prediction)
            1.0  # it is one even if the schema does not match (we introduce tuple constraints for this)
        """
        target = np.array(target)
        prediction = np.array(prediction)

        sum_cell_match = np.sum(np.isin(prediction, target))
        return round(sum_cell_match / prediction.size, 3)
