from collections import Counter

from .abstract_metric import AbstractMetric


class TupleConstraintTag(AbstractMetric):
    def evaluate_single_no_special_case(self,
                                        target: list[list | str | float],
                                        prediction: list[list | str | float]
                                        ) -> float | str:
        """
        check the ratio between the cardinality of the target tuples and the prediction.
        return a score between 0 and 1.
        Example:
            >> target = [['a', 'b'], ['c', 'd']]
            >> prediction = [['a', 'b'], ['c', 'd']]
            >> 1.0

            >> target = [['a', 'b'], ['c', 'd']]
            >> prediction = [['a', 'b'], ['a', 'b'], ['c', 'd']]
            >> 0.5

        :param target: target table to be compared with prediction table
        :param prediction: prediction table to be compared with target table
        :return: score between [0, 1]
            * 0 indicates NONE of the cardinality values are the same in prediction.
            * 1 indicates ALL the cardinality values are the same in prediction.

        """
        target = [tuple(t) for t in target]
        prediction = [tuple(t) for t in prediction]

        count_targ_dict = Counter(target)
        count_pred_dict = Counter(prediction)

        cardinality = [count_pred_dict[key] == count
                       for key, count in count_targ_dict.items()]

        return round(sum(cardinality) / len(cardinality), 3)
