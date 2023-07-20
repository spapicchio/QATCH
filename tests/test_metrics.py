import time

import numpy as np
import pytest

from metrics.cell_precision_tag import CellPrecisionTag
from metrics.cell_recall_tag import CellRecallTag
from metrics.tuple_cardinality_tag import TupleCardinalityTag
from metrics.tuple_constraint_tag import TupleConstraintTag
from metrics.tuple_order_tag import TupleOrderTag


class TestCellPrecisionTag:
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


class TestCellRecallTag:
    @pytest.fixture
    def instance(self):
        return CellRecallTag()

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

    def test_empty_tables(self, instance):
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
        assert result == round(1 / len(target[0]), 3)

        prediction = [[18], [18]]
        target = [['False', 'BC', 18, 0.8, 'AB', 'CA', 99, 5.4796861662000005, 'windows', 'Paid'],
                  ['True', 'BC', 18, 0.8, 'AB', 'CA', 157, 9.7706304446, 'linux', 'Paid']]
        result = instance.evaluate_single_test_metric(target, prediction)
        # 14 distinct elements in target
        assert result == round(2 / 20, 3)


class TestTupleCardinalityTag:
    @pytest.fixture
    def instance(self):
        return TupleCardinalityTag()

    def test_target_greater_than_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        prediction = [['a', 'b']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == round(1 / 3, 3)

    def test_prediction_greater_than_target(self, instance):
        target = [['a', 'b']]
        prediction = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == round(1 / 3, 3)

    def test_equal_target_and_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_empty_target_and_prediction(self, instance):
        target = []
        prediction = []
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_empty_prediction_with_non_empty_target(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = []
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0

    def test_non_empty_prediction_with_empty_target(self, instance):
        target = []
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0


class TestTupleConstraintTag:
    @pytest.fixture
    def instance(self):
        return TupleConstraintTag()

    def test_no_matching_tuples(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['x', 'y'], ['z', 'w']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'y'], ['c', 'w']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0

    def test_all_matching_tuples(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_partial_matching_tuples(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['a', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.5

    def test_empty_tables(self, instance):
        target = []
        prediction = []
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_single_tuple_table(self, instance):
        target = [['a', 'b']]
        prediction = [['a', 'b']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_no_tuples_in_prediction(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = []
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0

    def test_no_tuples_in_target(self, instance):
        target = []
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.0

    def test_duplicate_tuples_in_target(self, instance):
        target = [['a', 'b'], ['a', 'b'], ['c', 'd']]
        prediction = [['a', 'b'], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 0.5

    def test_all_matching_tuples_diff_order(self, instance):
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['c', 'd'], ['a', 'b']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0
        target = [['a', 'b'], ['c', 'd']]
        prediction = [['d', 'c'], ['b', 'a']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_special_case(self, instance):
        target = [[1, 2], ['c', 'd']]
        prediction = [[1, 2], ['c', 'd']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0
        target = [[1, 2], ['c', 'd']]
        prediction = [[1, 2], ['c', 'd'], ['a', 'b']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0
        target = [[1, 2], ['c', 'd']]
        prediction = [[1, 2], ['d', 'c'], ['a', 'b']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0
        target = [[1, 2], [1, 'd']]
        prediction = [[1, 2], ['d', 1], ['a', 'b']]
        result = instance.evaluate_single_test_metric(target, prediction)
        assert result == 1.0

    def test_evaluate_single_no_special_case_time(self, instance):
        target = np.empty((60000, 10), dtype=str).tolist()
        prediction = np.empty((60000, 10), dtype=str).tolist()
        start = time.time()
        _ = instance.evaluate_single_no_special_case(target, prediction)
        end = time.time()
        time_spent = round((end - start) / 60, 5)
        print(time_spent)
        assert time_spent < 0.001


class TestTupleOrderTag:
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
