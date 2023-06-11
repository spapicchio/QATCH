from tqdm import tqdm

from .cell_precision_tag import CellPrecisionTag
from .cell_recall_tag import CellRecallTag
from .tuple_cardinality_tag import TupleCardinalityTag
from .tuple_constraint_tag import TupleConstraintTag
from .tuple_order_tag import TupleOrderTag


class MetricEvaluator:
    def __init__(self, metrics: list[str] | str):
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.tags_generator = {
            'cell_precision': CellPrecisionTag,
            'cell_recall': CellRecallTag,
            'tuple_cardinality': TupleCardinalityTag,
            'tuple_constraint': TupleConstraintTag,
            'tuple_order': TupleOrderTag,
        }

    def evaluate(self,
                 targets: list[list[list]],
                 predictions: list[list[list]]
                 ) -> dict[str, list[float]]:
        metric2results = dict()
        for metric in tqdm(self.metrics, desc='Evaluating metrics'):
            generator = self.tags_generator[metric]()
            metric2results[metric] = generator.evaluate_tests(targets, predictions)
        return metric2results

    def evaluate_with_df(self, df, target: str, predictions: str):
        tqdm.pandas(desc='Evaluating metrics')
        for metric in tqdm(self.metrics, desc='Evaluating metrics'):
            generator = self.tags_generator[metric]()
            df.loc[:, metric] = df.progress_apply(
                lambda r: generator.evaluate_single_test_metric(r[target], r[predictions]),
                axis=1)
        return df
