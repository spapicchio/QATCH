import time

import numpy as np
import pytest

from qatch.evaluate_dataset.metrics_evaluators import TupleCardinality


class TestTupleCardinality:
    @pytest.fixture
    def instance(self):
        return TupleCardinality()

    def test_target_greater_than_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        prediction = [['a', 'b']]
        result = instance.run_metric(target, prediction)
        assert result == round(1 / 3, 3)

    def test_prediction_greater_than_target(self, instance):
        target = [['a', 'b']]
        prediction = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        result = instance.run_metric(target, prediction)
        assert result == round(1 / 3, 3)

    def test_equal_target_and_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_empty_target_and_prediction(self, instance):
        target = []
        prediction = []
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_empty_prediction_with_non_empty_target(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = []
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_non_empty_prediction_with_empty_target(self, instance):
        target = []
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_tuple_cardinality_time(self, instance):
        target = np.random.rand(20, 1000)
        prediction = np.random.rand(20, 1000)
        start_time = time.time()
        instance.run_metric(list(target), list(prediction))
        assert start_time - time.time() < 0.05

