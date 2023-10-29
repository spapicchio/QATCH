import math
import time

import numpy as np
import pytest

from qatch.metrics.abstract_metric import AbstractMetric


class TestNormalizeCell:
    class ConcreteMetric(AbstractMetric):
        def evaluate_single_no_special_case(self, target, prediction):
            return 42.0

    @pytest.fixture
    def abstract_metric_instance(self):
        return self.ConcreteMetric()

    def test_evaluate_single_special_case_empty_lists(self, abstract_metric_instance):
        target = []
        prediction = []
        result = abstract_metric_instance.evaluate_single_special_case(target, prediction)
        assert result == 1.0

    def test_evaluate_single_special_case_empty_prediction(self, abstract_metric_instance):
        target = [[1, 2], [3, 4]]
        prediction = []
        result = abstract_metric_instance.evaluate_single_special_case(target, prediction)
        assert result == 0.0

    def test_evaluate_single_special_case_empty_target(self, abstract_metric_instance):
        target = []
        prediction = [[1, 2], [3, 4]]
        result = abstract_metric_instance.evaluate_single_special_case(target, prediction)
        assert result == 0.0

    def test_evaluate_single_test_metric(self, abstract_metric_instance):
        target = [[1, 2], [3, 4]]
        prediction = [[5, 6], [7, 8]]
        result = abstract_metric_instance.evaluate_single_test_metric(target, prediction)
        assert result == 42.0

    def test_evaluate_tests(self, abstract_metric_instance):
        targets = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        predictions = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        results = abstract_metric_instance.evaluate_tests(targets, predictions)
        assert results == [42.0, 42.0]

    def test_normalize_cell(self, abstract_metric_instance):
        assert abstract_metric_instance.normalize_cell(42) == '42'
        assert abstract_metric_instance.normalize_cell('42') == '42.0'
        assert abstract_metric_instance.normalize_cell(3.14159) == '3.14'
        assert abstract_metric_instance.normalize_cell('HELLO\n ') == 'hello'
        assert abstract_metric_instance.normalize_cell(None) == 'None'
        assert abstract_metric_instance.normalize_cell(math.nan) == 'None'
        assert abstract_metric_instance.normalize_cell("abc123") == "abc123"
        assert abstract_metric_instance.normalize_cell(True) is True

    def test_normalize_time(self, abstract_metric_instance):
        target = np.random.rand(20, 1000)
        prediction = np.random.rand(20, 1000)
        start_time = time.time()
        abstract_metric_instance.evaluate_single_test_metric(list(target), list(prediction))
        assert start_time - time.time() < 0.5

# TODO move it in chatgpt tests
# class TestCheckChatGPTResult:
#     def test_none_input(self):
#         prediction = None
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result is None
#
#     def test_single_none_input(self):
#         prediction = [None]
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result is None
#
#     def test_string_input(self):
#         prediction = "[[1, 2], [3, 4]]"
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         expected = [[1, 2], [3, 4]]
#         assert result == expected
#
#     def test_invalid_string_input(self):
#         prediction = "[1, 2, [3, 4]]"
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result is None
#
#     def test_multi_dimensional_array_input(self):
#         prediction = [[[1], [2]], [[3], [4]]]
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         expected = [[1, 2], [3, 4]]
#         assert result == expected
#         prediction = [[[1, 2], [3, 4]]]
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         expected = [[1, 2], [3, 4]]
#         assert result == expected
#         prediction = [[[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]]]
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result == [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]
#
#     def test_empty_array_input(self):
#         prediction = []
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result == []
#         prediction = '[]'
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result == []
#
#     def test_scalar_input(self):
#         prediction = 5
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result == [[5]]
#
#     def test_presence_header(self):
#         prediction = "[['poisonous', '[H] class'], ['flat', '[H] capshape'], " \
#                      "['smooth', '[H] capsurface'], ['white', '[H] capcolor']," \
#                      " ['bruises', '[H]"
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         # the space is normalized later
#         assert result == [['poisonous '], ['flat '], ['smooth '], ['white ']]
#
#         prediction = "[['poisonous', '[H] class'], ['flat', '[H] capshape'], " \
#                      "['smooth', '[H] capsurface'], ['white', '[H] capcolor']"
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         # the space is normalized later
#         assert result == [['poisonous '], ['flat '], ['smooth '], ['white ']]
#         prediction = [['Simone', '[H] Name'], ['Papicchio', '[H] Surname'], ['25', '[H] Age']]
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result == [['Simone'], ['Papicchio'], ['25']]
#
#     def test_input_interrupt(self):
#         prediction = "[['averagebatterylife', 'devicetype', 'strapmaterial'], " \
#                      "[14, 'Smartwatch', 'Elastomer'],"
#         result = AbstractMetric.check_chatgpt_result(prediction)
#
#         assert result == [['averagebatterylife', 'devicetype', 'strapmaterial'],
#                           ['14', 'Smartwatch', 'Elastomer']]
#
#         prediction = "[['averagebatterylife', 'devicetype', 'strapmaterial,"
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result is None
#
#     def test_header_with_numbers(self):
#         prediction = "['White', '[H] race'], ['Female', '[H] sex'], [40, '[H] hoursperweek']"
#         result = AbstractMetric.check_chatgpt_result(prediction)
#         assert result == [['White '], ['Female '], ['40']]
