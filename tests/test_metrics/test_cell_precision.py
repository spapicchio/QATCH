import time

import numpy as np
import pytest

from qatch.metrics.cell_precision_tag import CellPrecisionTag


class TestCellPrecision:
    @pytest.fixture
    def instance(self):
        return CellPrecisionTag()

    def test_no_matching_cells(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['x', 'y'], ['z', 'w']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0

    def test_all_matching_cells(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_partial_matching_cells(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'x'], ['y', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.5

    def test_empty_values(self, instance):
        target = []
        prediction = []
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0
        target = '[]'
        prediction = '[]'
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_single_cell_table(self, instance):
        target = [['a']]
        prediction = [['a']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_no_cells_in_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = []
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0

    def test_no_cells_in_target(self, instance):
        target = []
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0

    def test_duplicate_cells_in_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'a'], ['b', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_special_chatgpt_case(self, instance):
        target = [['573585', '10/31/2019', '22282', '12-Egg-House-Painted-Wood',
                   35.83, 2, 14585.0, 'United-Kingdom']]
        prediction = [['573585']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_cell_precision_time(self, instance):
        target = np.random.rand(20, 1000)
        prediction = np.random.rand(20, 1000)
        start_time = time.time()
        instance.evaluate_single_test_metric(list(target), list(prediction))
        assert start_time - time.time() < 0.05
