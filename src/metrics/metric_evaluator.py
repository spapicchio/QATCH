from .cell_precision_tag import CellPrecisionTag
from .cell_recall_tag import CellRecallTag
from .row_cardinality_tag import RowCardinalityTag
from .tuple_cardinality_tag import TupleCardinalityTag
from .tuple_order_tag import TupleOrderTag


class MetricEvaluator:
    def __init__(self, metrics: list[str] | str):
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.tags_generator = {
            'cell_precision': CellPrecisionTag,
            'cell_recall': CellRecallTag,
            'row_cardinality': RowCardinalityTag,
            'tuple_cardinality': TupleCardinalityTag,
            'tuple_order': TupleOrderTag,
        }

    def evaluate(self,
                 targets: list[list[list]],
                 predictions: list[list[list]]
                 ) -> dict[str, list[float]]:
        metric2results = dict()
        for metric in self.metrics:
            generator = self.tags_generator[metric]()
            metric2results[metric] = generator.evaluate_tests(targets, predictions)

        return metric2results
