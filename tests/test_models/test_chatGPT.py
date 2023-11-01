import json
from unittest.mock import patch

import openai
import pandas as pd
import pytest

from qatch.models.chatgpt import ChatGPT_QA, ChatGPT_SP
from qatch.models.utils import _normalize_output_for_QA


class TestChatGPT:
    @pytest.fixture
    def chatgpt_instance_QA(self):
        with open('test_models/credentials.json', 'r') as f:
            credentials = json.load(f)
        # read credentials.json and get the api key
        model = ChatGPT_QA(api_key=credentials['api_key_chatgpt'],
                           api_org=credentials['api_org_chatgpt'])
        return model

    @pytest.fixture
    def chatgpt_instance_SP(self):
        with open('test_models/credentials.json', 'r') as f:
            credentials = json.load(f)
        # read credentials.json and get the api key
        model = ChatGPT_SP(api_key=credentials['api_key_chatgpt'],
                           api_org=credentials['api_org_chatgpt'])
        return model

    @pytest.fixture
    def table(self):
        table = pd.DataFrame(
            {'Student ID': [24172, 281811],
             'Grade': [30, 22],
             'Phone Numbers': [3431223445, 3435227445]})
        return table

    # Test the process_input method
    def test_process_input_QA(self, chatgpt_instance_QA, table):
        query = "what are all the phone numbers?"
        tbl_name = "Sample Table"
        result = chatgpt_instance_QA.process_input(table, query, tbl_name)

        assert result is not None
        assert isinstance(result, dict)
        assert 'role' in result
        assert 'content' in result

    def test_process_input_return_none_QA(self, chatgpt_instance_QA, table):
        # return None if the table is too large to be processed
        table = pd.concat([table] * 250)
        query = "what are all the phone numbers?"
        tbl_name = "Sample Table"
        result = chatgpt_instance_QA.process_input(table, query, tbl_name)
        assert result is None

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

    # Test case for successful API response
    def test_predict_input_successful_response(self, chatgpt_instance_QA):
        model_input = "some input"
        table = "some table"
        with patch.object(chatgpt_instance_QA, '_predict_with_api', return_value=[["result1", "result2"]]):
            result = chatgpt_instance_QA.predict_input(model_input, table)
        assert result == [["result1", "result2"]]

    # Test case for API error (AuthenticationError)
    @pytest.mark.parametrize("api_error, expected_result", [
        # invalid request is when the table is too large
        # only in this case, the result is None because the API does not return anything
        (openai.error.InvalidRequestError, [None]),
        (openai.error.RateLimitError, ['nan']),
        (openai.error.APIError, ['nan']),
        (openai.error.ServiceUnavailableError, ['nan']),
        (openai.error.Timeout, ['nan']),
    ])
    def test_predict_input_API_ERROR(self, chatgpt_instance_QA, api_error, expected_result):
        model_input = "some input"
        table = "some table"
        with patch.object(chatgpt_instance_QA, '_predict_with_api', side_effect=api_error('', '')):
            result = chatgpt_instance_QA.predict_input(model_input, table)
        assert result == expected_result

    def test_predict_QA(self, chatgpt_instance_QA, table):
        query = "what are all the phone numbers?"
        tbl_name = "Sample Table"
        result = chatgpt_instance_QA.predict(table, query, tbl_name)
        assert result == [[3431223445], [3435227445]]

    def test_predict_SP(self, chatgpt_instance_SP, table):
        query = "what are all the phone numbers?"
        tbl_name = "Sample Table"
        result = chatgpt_instance_SP.predict(table, query, tbl_name)
        assert result == 'SELECT "Phone Numbers" FROM "Sample Table"'
