import pandas as pd
import pytest

from qatch.models.utils import _normalize_output_for_QA


class TestChatGPT:
    @pytest.fixture
    def table(self):
        table = pd.DataFrame(
            {'Student ID': [24172, 281811],
             'Grade': [30, 22],
             'Phone Numbers': [3431223445, 3435227445]})
        return table

    @pytest.mark.parametrize("prediction, expected_result", [
        (None, None),  # if the prediction is None, the output should be None
        ([None], None),
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),  # the string should be converted to a list
        ("[1, 2, [3, 4]]", None),  # string cannot be converted
        ("[[1], [2]], [[3], [4]]]", [[1], [2], [3], [4]]),  # multidimensional array with an error
        ("[[1], [2]], [[3], [4]]", [[1, 2], [3, 4]]),  # multidimensional array
        ("[[[1, 2], [3, 4]]]", [[1, 2], [3, 4]]),
        (" [[[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]]]", [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]),
        ("[]", []),  # empty array as output
        ("5", [[5]]),  # scalar as output
        ("[['poisonous'], ['flat', ", [['poisonous']]),  # Not complete output
        ("[['poisonous'], ['flat'], [blac...", [['poisonous'], ['flat']]),  # Not complete output
        ("[['poisonous', '[H] class'], ['flat', '[H] capshape']]", [['poisonous'], ['flat']]),  # presence of [H]
        ("[['poisonous', '[H] class'], ['flat', ", [['poisonous ']]),  # presence of [H]
        # H with numbers
        ("['White', '[H] race'], ['Female', '[H] sex'], [40, '[H] hoursperweek']", [['White'], ['Female'], ['40']]),
        ("['White', '[H] race'], ['Female', '[H] s", [['White ']]),
        ("[['averagebatterylife', 'devicetype', 'strapmaterial,", None),  # not completed
        ("[['averagebatterylife'], [12]]", [['averagebatterylife'], ['12']]),  # numbers converted str
    ])
    def test_normalize_output_for_QA(self, prediction: str, expected_result):
        output = _normalize_output_for_QA(prediction)
        assert output == expected_result
