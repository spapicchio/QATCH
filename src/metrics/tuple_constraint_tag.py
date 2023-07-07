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
        target, prediction = self.normalize_target_prediction(target, prediction)

        target = map(tuple, target)
        prediction = map(tuple, prediction)

        count_targ_dict = Counter(target)
        count_pred_dict = Counter(prediction)

        cardinality = [count_pred_dict[key] == count
                       for key, count in count_targ_dict.items()]

        return round(sum(cardinality) / len(cardinality), 3)

    def normalize_target_prediction(self, target, prediction):
        """
        Check cell by cell whether the cell is present but with different order.
        If the cell is present but with different order,
        then the cell is replaced with the target cell.
        :example:
            target = [['a', 'b'], ['c', 'd']]
            prediction = [['d', 'c'], ['b', 'a']]
            return  [['a', 'b'], ['c', 'd']],  [['a', 'b'], ['c', 'd']]
        """
        prediction = map(set, prediction)
        map_target = list(map(set, target))
        new_prediction = []
        for pred_row in prediction:
            if pred_row in map_target:
                element = target[map_target.index(pred_row)]
                new_prediction.append(element)
            else:
                new_prediction.append(pred_row)
        return target, new_prediction
