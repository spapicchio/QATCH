import json
import os

import pandas as pd
import pytest

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from qatch.models import LLama2_QA, LLama2_SP


class TestLLama2:
    @pytest.fixture
    def llama2_instance_QA(self):
        with open('test_models/credentials.json', 'r') as f:
            credentials = json.load(f)
        return LLama2_QA("meta-llama/Llama-2-7b-chat-hf",
                         hugging_face_token=credentials['hugging_face_token'])

    @pytest.fixture
    def llama2_instance_SP(self):
        with open('test_models/credentials.json', 'r') as f:
            credentials = json.load(f)
        return LLama2_SP("meta-llama/Llama-2-7b-chat-hf",
                         hugging_face_token=credentials['hugging_face_token'])

    @pytest.fixture
    def table(self):
        table = pd.DataFrame(
            {'Student ID': [24172, 281811],
             'Grade': [30, 22],
             'Phone Numbers': [3431223445, 3435227445]})
        return table

    def test_predict_QA(self, llama2_instance_QA, table):
        query = "what are all the phone numbers?"
        tbl_name = "Sample Table"
        result = llama2_instance_QA.predict(table, query, tbl_name)
        assert result == [[3431223445], [3435227445]]

    def test_predict_SP(self, llama2_instance_SP, table):
        query = "what are all the phone numbers?"
        tbl_name = "Sample Table"
        result = llama2_instance_SP.predict(table, query, tbl_name)
        assert result == 'SELECT "Phone Numbers" FROM "Sample Table"'
