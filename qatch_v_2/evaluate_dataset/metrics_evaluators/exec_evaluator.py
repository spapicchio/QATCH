from __future__ import annotations

from .base_evaluator import BaseEvaluator


class ExecutionAccuracy(BaseEvaluator):
    @property
    def metric_name(self):
        return 'execution_accuracy'

    def run_metric(self, target: list[list], prediction: list[list]) -> float | int:
        """
        Calculates the execution accuracy between the target and prediction.

        This function compares a set of target rows with the prediction by creating a unique representation of data sequence.
        If the matching rows are not found, it returns 0.0 else it continues to check other rows.
        If all rows match perfectly returns 1.0, else it returns 0.0.

        Args:
            target (list[list]): The target list of lists, each representing a unique sequence of data.
            prediction (list[list]): The prediction list of lists, each representing a matched sequence of data.

        Returns:
            float | int: Returns 1.0 if all rows match perfectly, otherwise returns 0.0.

        Note:
            - The function uses the function sort_with_different_types to sort data having different data types.
            - The metric is robust to different tuple order between predictions and target
            - The metric is robust to different projection order (Name, Surname) = (Surname, Name)
        """

        if len(target) != len(prediction):
            return 0.0

        gold_row_set = set()
        pred_row_set = set()

        for gold_row, predicted_row in zip(target, prediction):
            gold_row_set.add(tuple(sort_with_different_types(gold_row)))
            pred_row_set.add(tuple(sort_with_different_types(predicted_row)))

        while len(pred_row_set) > 0:
            pred_row = pred_row_set.pop()
            if pred_row in gold_row_set:
                gold_row_set.remove(pred_row)
            else:
                return 0.0

        return 1.0 if len(gold_row_set) == 0 else 0.0


def sort_key(x):
    """Transforms the input value into a tuple for consistent comparison.

    This method is primarily used as a key function for Python's built-in sorting.
    It transforms the raw input into a tuple that can be used for comparison across various types.
    None values are treated as smallest, followed by numerical types, and then all other types are converted to strings.

    Args:
        x : Variable of any data type.
            The data that needs to be transformed for sorting.

    Returns:
        tuple: A two-element tuple that consists of a priority indicator (int) and a transformed value (float or str).

    Note:
        - None is treated as smallest and assigned a priority of 0.
        - Numerical types (int and float) are assigned a priority of 1 and are uniformly represented as float.
        - All other types are converted to string and assigned a priority of 2.
        - This makes it possible to sort a list containing diverse types of elements.

    """
    if x is None:
        return 0, ''
    elif isinstance(x, (int, float)):
        return 1, float(x)
    else:
        return 2, str(x)


def sort_with_different_types(arr):
    sorted_arr = sorted(arr, key=sort_key)
    return sorted_arr
