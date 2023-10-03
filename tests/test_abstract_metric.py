import math

from metrics.abstract_metric import AbstractMetric


class TestNormalizeCell:
    def test_valid_integer(self):
        cell = 10
        result = AbstractMetric.normalize_cell(cell)
        assert result == '10'

    def test_valid_float(self):
        cell = 3.14
        result = AbstractMetric.normalize_cell(cell)
        assert result == '3'

    def test_valid_string_integer(self):
        cell = "42"
        result = AbstractMetric.normalize_cell(cell)
        assert result == '42'

    def test_valid_string_float(self):
        cell = "3.14159"
        result = AbstractMetric.normalize_cell(cell)
        assert result == '3'

    def test_string_with_whitespace_and_newlines(self):
        cell = "   hello\n"
        result = AbstractMetric.normalize_cell(cell)
        assert result == "hello"

    def test_string_with_non_numeric_characters(self):
        cell = "abc123"
        result = AbstractMetric.normalize_cell(cell)
        assert result == "abc123"

    def test_nan_value(self):
        cell = math.nan
        result = AbstractMetric.normalize_cell(cell)
        assert result == 'None'

    def test_non_string_non_nan_value(self):
        cell = True
        result = AbstractMetric.normalize_cell(cell)
        assert result


class TestCheckChatGPTResult:
    def test_none_input(self):
        prediction = None
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result is None

    def test_single_none_input(self):
        prediction = [None]
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result is None

    def test_string_input(self):
        prediction = "[[1, 2], [3, 4]]"
        result = AbstractMetric.check_chatgpt_result(prediction)
        expected = [[1, 2], [3, 4]]
        assert result == expected

    def test_invalid_string_input(self):
        prediction = "[1, 2, [3, 4]]"
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result is None

    def test_multi_dimensional_array_input(self):
        prediction = [[[1], [2]], [[3], [4]]]
        result = AbstractMetric.check_chatgpt_result(prediction)
        expected = [[1, 2], [3, 4]]
        assert result == expected
        prediction = [[[1, 2], [3, 4]]]
        result = AbstractMetric.check_chatgpt_result(prediction)
        expected = [[1, 2], [3, 4]]
        assert result == expected
        prediction = [[[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]]]
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result == [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]

    def test_empty_array_input(self):
        prediction = []
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result == []
        prediction = '[]'
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result == []

    def test_scalar_input(self):
        prediction = 5
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result == [[5]]

    def test_presence_header(self):
        prediction = "[['poisonous', '[H] class'], ['flat', '[H] capshape'], " \
                     "['smooth', '[H] capsurface'], ['white', '[H] capcolor']," \
                     " ['bruises', '[H]"
        result = AbstractMetric.check_chatgpt_result(prediction)
        # the space is normalized later
        assert result == [['poisonous '], ['flat '], ['smooth '], ['white ']]

        prediction = "[['poisonous', '[H] class'], ['flat', '[H] capshape'], " \
                     "['smooth', '[H] capsurface'], ['white', '[H] capcolor']"
        result = AbstractMetric.check_chatgpt_result(prediction)
        # the space is normalized later
        assert result == [['poisonous '], ['flat '], ['smooth '], ['white ']]
        prediction = [['Simone', '[H] Name'], ['Papicchio', '[H] Surname'], ['25', '[H] Age']]
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result == [['Simone'], ['Papicchio'], ['25']]

    def test_input_interrupt(self):
        prediction = "[['averagebatterylife', 'devicetype', 'strapmaterial'], " \
                     "[14, 'Smartwatch', 'Elastomer'],"
        result = AbstractMetric.check_chatgpt_result(prediction)

        assert result == [['averagebatterylife', 'devicetype', 'strapmaterial'],
                          ['14', 'Smartwatch', 'Elastomer']]

        prediction = "[['averagebatterylife', 'devicetype', 'strapmaterial,"
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result is None

    def test_header_with_numbers(self):
        prediction = "['White', '[H] race'], ['Female', '[H] sex'], [40, '[H] hoursperweek']"
        result = AbstractMetric.check_chatgpt_result(prediction)
        assert result == [['White '], ['Female '], ['40']]
