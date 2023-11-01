import json
from unittest.mock import patch

import openai
import pandas as pd
import pytest

from qatch.models.chatgpt import ChatGPT_QA


class TestAbstractChatGPT:
    @pytest.fixture
    def chatgpt_instance(self):
        with open('test_models/credentials.json', 'r') as f:
            credentials = json.load(f)
        # read credentials.json and get the api key
        model = ChatGPT_QA(api_key=credentials['api_key_chatgpt'],
                           api_org=credentials['api_org_chatgpt'])
        return model

    @pytest.fixture
    def table(self):
        table = pd.DataFrame(
            {'Student ID': [24172, 281811],
             'Grade': [30, 22],
             'Phone Numbers': [3431223445, 3435227445]})
        return table

    # Test case for successful API response
    def test_predict_input_successful_response(self, chatgpt_instance):
        model_input = "some input"
        table = "some table"
        with patch.object(chatgpt_instance, '_predict_with_api', return_value=[["result1", "result2"]]):
            result = chatgpt_instance.predict_input(model_input, table)
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
    def test_predict_input_API_ERROR(self, chatgpt_instance, api_error, expected_result):
        model_input = "some input"
        table = "some table"
        with patch.object(chatgpt_instance, '_predict_with_api', side_effect=api_error(message='',
                                                                                       param='')):
            result = chatgpt_instance.predict_input(model_input, table)
        assert result == expected_result

    def test_predict(self, chatgpt_instance, table):
        query = "what are all the phone numbers?"
        tbl_name = "Sample Table"
        result = chatgpt_instance.predict(table, query, tbl_name)
        assert result == [[3431223445], [3435227445]]


