from tqdm import tqdm

from .cell_precision_tag import CellPrecisionTag
from .cell_recall_tag import CellRecallTag
from .tuple_cardinality_tag import TupleCardinalityTag
from .tuple_constraint_tag import TupleConstraintTag
from .tuple_order_tag import TupleOrderTag


class MetricEvaluator:
    def __init__(self, metrics: list[str] | str | None):
        if metrics is None:
            metrics = ['cell_precision', 'cell_recall', 'tuple_cardinality',
                       'tuple_constraint', 'tuple_order']
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

    def evaluate_with_df(self, df, target: str, predictions: str, task='QA'):
        if task == 'QA':
            return self.evaluate_with_df_qa(df, target, predictions)
        elif task == 'SP':
            return self.evaluate_with_df_sp(df, target, predictions)
        else:
            raise ValueError(f'Unknown task: {task}')

    def evaluate_with_df_qa(self, df, target, predictions):
        tqdm.pandas(desc='Evaluating metrics')
        for metric in tqdm(self.metrics, desc='Evaluating metrics'):
            generator = self.tags_generator[metric]()
            if metric == 'tuple_order':
                df.loc[:, metric] = None
                mask = df['sql_tags'].str.contains('ORDERBY')
                df.loc[mask, metric] = df[mask].progress_apply(
                    lambda r: generator.evaluate_single_test_metric(r[target], r[predictions]),
                    axis=1)
            else:
                df.loc[:, metric] = df.progress_apply(
                    lambda r: generator.evaluate_single_test_metric(r[target], r[predictions]),
                    axis=1)
        return df

    def evaluate_with_df_sp(self, df, target: str, predictions: str):
        """
        the difference with QA is that if the target SQL and the Prediction SQL
        are equal, there is no need to run the evaluation over the SQL results.
        """
        tqdm.pandas(desc='Evaluating metrics')
        mask_order = df.sql_tags.str.contains('ORDERBY')
        mask_equal = df[predictions] == 'EQUAL'
        for metric in self.metrics:
            generator = self.tags_generator[metric]()
            df[metric] = None

            if metric != 'tuple_order':
                df.loc[mask_equal, metric] = 1
                df.loc[~mask_equal, metric] = df[~mask_equal].progress_apply(
                    lambda r: generator.evaluate_single_test_metric(r[target], r[predictions]),
                    axis=1)
            else:
                if any(mask_order):
                    # case where the tuple_order is called but no ORDERBY is present
                    df.loc[mask_equal & mask_order, metric] = 1
                    df.loc[~mask_equal & mask_order, metric] = df[~mask_equal & mask_order].progress_apply(
                        lambda r: generator.evaluate_single_test_metric(r[target], r[predictions]),
                        axis=1)
        return df
