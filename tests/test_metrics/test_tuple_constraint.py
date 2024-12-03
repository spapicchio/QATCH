import math
import time

import numpy as np
import pytest

from qatch.evaluate_dataset.metrics_evaluators import TupleConstraint


class TestTupleConstraintTag:
    @pytest.fixture
    def instance(self):
        return TupleConstraint()

    def test_no_matching_tuples(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['x', 'y'], ['z', 'w']]
        result = instance.run_metric(target, prediction)
        assert result == 0.0
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'y'], ['c', 'w']]
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_all_matching_tuples(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_partial_matching_tuples(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 0.5

    def test_empty_tables(self, instance):
        target = []
        prediction = []
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_single_tuple_table(self, instance):
        target = [['a', 'b']]
        prediction = [['a', 'b']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_no_tuples_in_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = []
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_no_tuples_in_target(self, instance):
        target = []
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_duplicate_tuples_in_target(self, instance):
        target = [['a', 'b'], ['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 0.5

    def test_all_matching_tuples_diff_order(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['c', 'd'], ['a', 'b']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['d', 'c'], ['b', 'a']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_special_case(self, instance):
        target = [[1, 2], ['c', 'd']]
        prediction = [[1, 2], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0
        target = [[1, 2], ['c', 'd']]
        prediction = [[1, 2], ['c', 'd'], ['a', 'b']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0
        target = [[1, 2], ['c', None, math.nan]]
        prediction = [[1, 2], [None, math.nan,  'c'], ['a', 'b']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0
        target = [[1, 2], [1, 'd']]
        prediction = [[2, 1], ['d', 1], ['a', 'b']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_evaluate_single_no_special_case_time(self, instance):
        target = np.random.rand(20, 1000).tolist()
        prediction = np.random.rand(20, 1000).tolist()
        start = time.time()
        _ = instance.evaluate_single_no_special_case(target, prediction)
        end = time.time()
        assert start - end < 0.001
