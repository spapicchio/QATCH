import pytest

from qatch.evaluate_dataset.metrics_evaluators import TupleOrder


class TestTupleOrder:
    @pytest.fixture
    def instance(self):
        return TupleOrder()

    def test_evaluate_opposite_direction(self, instance):
        target = [['apple', 'orange'], ['pear']]
        prediction = [['pear'], ['apple', 'orange']]
        score = instance.run_metric(target, prediction)
        assert score == 0.0
        target = [['a'], ['b'], [None]]
        prediction = [[None], ['b'], ['a']]
        score = instance.run_metric(target, prediction)
        assert score == 0.0

    def test_evaluate_empty_input(self, instance):
        target = []
        prediction = []
        score = instance.run_metric(target, prediction)
        assert score == 1.0

    def test_evaluate_same_order(self, instance):
        target = [['a'], ['b'], ['c']]
        prediction = [['a'], ['b'], ['c']]
        score = instance.run_metric(target, prediction)
        assert score == 1.0

    def test_evaluate_more_elements_pred(self, instance):
        target = [['a'], ['b'], ['c'], ['d']]
        prediction = [['c'], ['b'], ['e'], ['f']]
        # prediction after normalization  = [['c'], ['b']]
        score = instance.run_metric(target, prediction)
        assert score == 0.0

    def test_evaluate_more_elements_target(self, instance):
        target = [['a'], ['a'], ['a'], ['b'], ['c'], ['d']]
        prediction = [['a'], ['b'], ['b'], ['c'], ['e']]
        score = instance.run_metric(target, prediction)
        assert score == 1.0

    def test_evaluate_no_correlation(self, instance):
        target = [['a'], ['b'], ['c']]
        prediction = [['d'], ['e'], ['f']]
        score = instance.run_metric(target, prediction)
        assert score != 0.0 and score != 1.0
