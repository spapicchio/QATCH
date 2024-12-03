import math

import pytest

from qatch.evaluate_dataset.metrics_evaluators import ExecutionAccuracy


class TestCellRecall:
    @pytest.fixture
    def instance(self):
        return ExecutionAccuracy()

    def test_equal(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_different(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_equal_but_different_projection(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['b', 'a'], ['d', 'c']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_equal_but_different_tuple_order(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['d', 'c'], ['b', 'a']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_null_values(self, instance):
        target = [['a', None], ['c', 'd']]
        prediction = [['d', 'c'], [None, 'a']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_null_math_none(self, instance):
        target = [['a', math.nan], ['c', 'd']]
        prediction = [['d', 'c'], [math.nan, 'a']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0
