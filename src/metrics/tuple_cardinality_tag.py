from .abstract_metric import AbstractMetric


class TupleCardinalityTag(AbstractMetric):
    def evaluate_single_no_special_case(self,
                                        target: list[list],
                                        prediction: list[list]) -> float | str:
        """
         len(target) / len(prediction)
          or vice-versa based on max len (to have always between 0 and 1)
        Example:
            >> target:[ [a, b], [c, d], [c, d], [f, g]
            >> prediction: [ [a, b], [3, 2] ]
            >> 0.5

        :param target: target table to be compared with prediction table
        :param prediction: prediction table to be compared with target table
        :return: score between [0,1]
            * 0 indicates the target/prediction is zero and the other is not.
            * 1 indicates the target/prediction is the same size as the other.

        """
        if len(prediction) >= len(target):
            # in case we have more elements in the prediction than in the target
            return len(target) / len(prediction)

        # in case we have more elements in the target than in the prediction
        elif len(prediction) < len(target):
            return round(len(prediction) / len(target), 3)
