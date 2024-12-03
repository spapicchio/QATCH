import time

import numpy as np
import pytest

from qatch.evaluate_dataset.metrics_evaluators import CellRecall


class TestCellRecall:
    @pytest.fixture
    def instance(self):
        return CellRecall()

    def test_no_matching_cells(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['x', 'y'], ['z', 'w']]
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_all_matching_cells(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_partial_matching_cells(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'x'], ['y', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 0.5

    def test_empty_tables(self, instance):
        target = []
        prediction = []
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_single_cell_table(self, instance):
        target = [['a']]
        prediction = [['a']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_no_cells_in_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = []
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_no_cells_in_target(self, instance):
        target = []
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 0.0

    def test_duplicate_cells_in_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'a'], ['b', 'b'], ['c', 'd']]
        result = instance.run_metric(target, prediction)
        assert result == 1.0

    def test_special_chatgpt_case(self, instance):
        target = [['573585', '10/31/2019', '22282', '12-Egg-House-Painted-Wood',
                   35.83, 2, 14585.0, 'United-Kingdom']]
        prediction = [['573585']]
        result = instance.run_metric(target, prediction)
        assert result == round(1 / len(target[0]), 3)

        prediction = [[18], [18]]
        target = [
            ['False', 'BC', 18, 0.8, 'AB', 'CA', 99, 5.4796861662000005, 'windows', 'Paid'],
            ['True', 'BC', 18, 0.8, 'AB', 'CA', 157, 9.7706304446, 'linux', 'Paid']
        ]
        result = instance.run_metric(target, prediction)
        # 14 distinct elements in target
        assert result == round(1 / 14, 3)

    def test_cell_recall_time(self, instance):
        target = np.random.rand(20, 1000)
        prediction = np.random.rand(20, 1000)
        start_time = time.time()
        instance.run_metric(list(target), list(prediction))
        assert start_time - time.time() < 0.05
