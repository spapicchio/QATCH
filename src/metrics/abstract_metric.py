from abc import ABC, abstractmethod


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
                                    ) -> float | str:
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
        if prediction is None:
            return None

        # normalize target and prediction
        target = [
            [str(cell).replace('\n', '').strip().lower() for cell in row]
            for row in target]

        prediction = [[str(cell).strip().lower() for cell in row]
                      for row in prediction]

        if len(target) == 0 or len(prediction) == 0:
            return self.evaluate_single_special_case(target, prediction)
        else:
            return self.evaluate_single_no_special_case(target, prediction)

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
