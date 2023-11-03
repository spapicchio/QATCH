import math
import re
from abc import ABC, abstractmethod

import numpy as np


class AbstractMetric(ABC):
    """
    An abstract base class for defining evaluation metrics for test predictions.
    Subclasses must implement the abstract method `evaluate_single_no_special_case`.
    """

    def evaluate_tests(self,
                       targets: list[list[list]],
                       predictions: list[list[list]]
                       ) -> list[float]:
        """
        Evaluates multiple tests using the implemented metric.
        :param list[list[list]] targets: A list of target values for multiple tests.
        :param list[list[list]] predictions: A list of predicted values for multiple tests.
        :return: A list of evaluation scores for each test.
        """
        return list(map(self.evaluate_single_test_metric, targets, predictions))

    def evaluate_single_test_metric(self,
                                    target: list[list],
                                    prediction: list[list] | None
                                    ) -> float:
        """
        Evaluates a single test using the implemented metric.

        :param list[list] target: The target values for a test.
        :param list[list] prediction: The predicted values for a test.

        :return: The evaluation score for the test.
        """
        if prediction is None:
            # in case the model was not able to predict anything
            return 0.0

        target = [] if target == '[]' else target
        prediction = [] if prediction == '[]' else prediction

        # normalize target and prediction
        target = [list(map(self.normalize_cell, row)) for row in target]
        prediction = [list(map(self.normalize_cell, row)) for row in prediction]

        if len(target) == 0 or len(prediction) == 0:
            return self.evaluate_single_special_case(target, prediction)

        else:
            return self.evaluate_single_no_special_case(target, prediction)

    @staticmethod
    def normalize_cell(cell):
        """
        Normalizes a cell value for comparison.
        All the cell values are transformed in string.
        The numbers are rounded to 2 decimal places.

        :param cell: The cell value to normalize.
        :return: The normalized cell value.
        """
        if cell is None:
            return "None"
        elif isinstance(cell, bool):
            return cell
        elif not isinstance(cell, str) and math.isnan(cell):
            return 'None'
        elif isinstance(cell, (np.float_, np.int_)):
            cell = str(round(cell, 2))
        elif isinstance(cell, (float, int)):
            cell = str(round(cell, 2))
        elif isinstance(cell, str) and re.match(r'^-?\d+(?:\.\d+)?$', cell):
            # number as string
            # round only if it has decimal places
            if '.' in cell:
                cell = str(round(float(cell), 2))
            else:
                cell = str(int(cell))
        else:
            # string with no numbers
            cell = cell.replace('\n', '').strip().lower()
        return cell

    @abstractmethod
    def evaluate_single_no_special_case(self,
                                        target: list[list],
                                        prediction: list[list]
                                        ) -> float:
        """
        Abstract method to evaluate a single test when there are no special cases.
        This method must be implemented by subclasses.
        :param list[list] target: The target values for a test.
        :param list[list] prediction: The predicted values for a test.
        :return: The evaluation score for the test.
        """

        raise NotImplementedError

    @staticmethod
    def evaluate_single_special_case(target: list[list],
                                     prediction: list[list]
                                     ) -> float:
        """
        Evaluates a single test when the target or the prediction is zero.
        :param list[list] target: The target values for a test.
        :param list[list] prediction: The predicted values for a test.
        :return: The evaluation score for the test.
        """

        if len(target) == 0 and len(prediction) == 0:
            return 1.0
        if len(prediction) == 0 and len(target) > 0:
            return 0.0
        if len(target) == 0 and len(prediction) > 0:
            return 0.0
