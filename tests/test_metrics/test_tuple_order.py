import time

import numpy as np
import pytest

from qatch.metrics.tuple_order_tag import TupleOrderTag


class TestTupleOrder:
    @pytest.fixture
    def instance(self):
        return TupleOrderTag()

    def test_evaluate_opposite_direction(self, instance):
        target = [['apple', 'orange'], ['pear']]
        prediction = [['pear'], ['apple', 'orange']]
        score = instance.evaluate_single_test_metric(target, prediction)
        assert score == 0.0
        target = [['a'], ['b'], ['c']]
        prediction = [['c'], ['b'], ['a']]
        score = instance.evaluate_single_test_metric(target, prediction)
        assert score == 0.0

    def test_evaluate_empty_input(self, instance):
        target = []
        prediction = []
        score = instance.evaluate_single_test_metric(target, prediction)
        assert score == 1.0

    def test_evaluate_same_order(self, instance):
        target = [['a'], ['b'], ['c']]
        prediction = [['a'], ['b'], ['c']]
        score = instance.evaluate_single_test_metric(target, prediction)
        assert score == 1.0

    def test_evaluate_more_elements_pred(self, instance):
        target = [['a'], ['b'], ['c'], ['d']]
        prediction = [['c'], ['b'], ['e'], ['f']]
        # prediction after normalization  = [['c'], ['b']]
        score = instance.evaluate_single_test_metric(target, prediction)
        assert score == 0.0

    def test_evaluate_more_elements_target(self, instance):
        target = [['a'], ['a'], ['a'], ['b'], ['c'], ['d']]
        prediction = [['a'], ['b'], ['b'], ['c'], ['e']]
        score = instance.evaluate_single_test_metric(target, prediction)
        assert score == 1.0

    def test_evaluate_no_correlation(self, instance):
        target = [['a'], ['b'], ['c']]
        prediction = [['d'], ['e'], ['f']]
        score = instance.evaluate_single_test_metric(target, prediction)
        assert score != 0.0 and score != 1.0

    def test_tuple_order_time(self, instance):
        target = np.random.rand(200, 1000)
        prediction = np.random.rand(200, 1000)
        start_time = time.time()
        instance.evaluate_single_test_metric(list(target), list(prediction))
        assert start_time - time.time() < 0.05
