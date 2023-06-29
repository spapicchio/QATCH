import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractMetric(ABC):
    def evaluate_tests(self,
                       targets: list[list[list]],
                       predictions: list[list[list]]
                       ) -> list[float | str]:
        """
        Evaluates the metric for each test.
        The type of prediction and target is given by the linearized table output
        i.e. one list for each row of the table and one list for the values in the row
            target: [['a', 'b'], ['c', 'd']]
            prediction: [['a', 'b'], ['c', 'd']]

        :param targets: values to be compared with predictions
        :param predictions: predictions to be compared with target
        :return: list of metric results (float or str)
        """
        return list(map(self.evaluate_single_test_metric, targets, predictions))

    def evaluate_single_test_metric(self,
                                    target: list[list],
                                    prediction: list[list]
                                    ) -> float | str | None:
        """
        Evaluates the metric for a single test.
        The type of prediction and target is given by the linearized table output
        i.e. one list for each row of the table and one list for the values in the row
            target: [['a', 'b'], ['c', 'd']]
            prediction: [['a', 'b'], ['c', 'd']]

        :param target: target table to be compared with prediction table
        :param prediction: prediction table to be compared with target table
        :return: the metric result (float or str)
        """
        prediction = self.check_chatgpt_result(prediction)
        if prediction is None:
            return 0.0

        # normalize target and prediction
        target = [list(map(self.normalize_cell, row)) for row in target]
        prediction = [list(map(self.normalize_cell, row)) for row in prediction]

        if len(target) == 0 or len(prediction) == 0:
            return self.evaluate_single_special_case(target, prediction)
        else:
            return self.evaluate_single_no_special_case(target, prediction)

    @staticmethod
    def normalize_cell(cell):
        if isinstance(cell, float):
            cell = round(cell)
        elif isinstance(cell, str) and re.match(r'^-?\d+(?:\.\d+)?$', cell):
            cell = round(float(cell), 1)
        return str(cell).replace('\n', '').strip().lower()

    @staticmethod
    def check_chatgpt_result(prediction) -> list[list[Any]] | None:
        if prediction is None:
            return None
        if prediction == [None]:
            return None

        if isinstance(prediction, str):
            try:
                while len(prediction) > 0 and prediction[0] == '[':
                    prediction = prediction[1:]
                while len(prediction) > 0 and prediction[-1] != ']':
                    prediction = prediction[:-1]
                while len(prediction) > 0 and prediction[-1] == ']':
                    prediction = prediction[:-1]
                if len(prediction) == 0:
                    return None
                prediction = f'[[{prediction}]]'

                prediction = eval(prediction)
                if isinstance(prediction, tuple):
                    new_pred = []
                    [new_pred.extend(p) for p in prediction]
                    prediction = new_pred
            except NameError:
                return None
            except SyntaxError:
                return None
            except TypeError:
                return None

        try:
            # may fail because len of the inside array are not equal
            prediction = np.array(prediction)
        except ValueError:
            return None

        if len(prediction.shape) > 2:
            while 1 in prediction.shape:
                axes = [ax for ax, x in enumerate(prediction.shape) if x == 1]
                prediction = np.squeeze(prediction, axis=axes[0])

            if prediction.shape == ():
                return [[prediction.tolist()]]
            elif len(prediction.shape) == 1:
                return [[x] for x in prediction]
            else:
                return prediction.tolist()

        elif len(prediction.shape) == 1:
            return [[x] for x in prediction]

        elif len(prediction.shape) == 0:
            return [[prediction.tolist()]]
        return prediction.tolist()

    @abstractmethod
    def evaluate_single_no_special_case(self,
                                        target: list[list],
                                        prediction: list[list]
                                        ) -> float:
        """
        :param target: target table to be compared with prediction table
        :param prediction: prediction table to be compared with target table
        :return: the metric result (float or str)
        """
        raise NotImplementedError

    def evaluate_single_special_case(self,
                                     target: list[list],
                                     prediction: list[list]
                                     ) -> float:
        """
        special case where target | prediction is empty.

        :param target: target table to be compared with prediction table
        :param prediction: prediction table to be compared with target table
        :return: the metric result or None if not a special case
        """
        if len(target) == 0 and len(prediction) == 0:
            return 1.0
        if len(prediction) == 0 and len(target) > 0:
            return 0.0
        if len(target) == 0 and len(prediction) > 0:
            return 0.0
