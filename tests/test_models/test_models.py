import pandas as pd
import pytest

from qatch.models import Omnitab, Tapas, Tapex


@pytest.fixture
def table():
    table = pd.DataFrame(
        {'Student ID': [24172, 281811],
         'Grade': [30, 22],
         'Phone Numbers': [3431223445, 3435227445]})
    return table


class TestOmnitabTapasTapex:
    @pytest.fixture
    def omnitab_instance(self):
        model = Omnitab(model_name='neulab/omnitab-large-finetuned-wtq',
                        force_cpu=True)
        return model

    @pytest.fixture
    def tapas_instance(self):
        model = Tapas(model_name='google/tapas-large-finetuned-wtq',
                      force_cpu=True)
        return model

    @pytest.fixture
    def tapex_instance(self):
        model = Tapex(model_name='microsoft/tapex-large-finetuned-wtq',
                      force_cpu=True)
        return model

    def test_process_input(self, omnitab_instance, tapas_instance, tapex_instance, table):
        query = "what are all the phone numbers?"
        tbl_name = "table"
        for model in [omnitab_instance, tapas_instance, tapex_instance]:
            model_input = model.process_input(table, query, tbl_name)
            # Assert
            assert model_input is not None

    def test_process_input_too_long(self, omnitab_instance, tapas_instance, tapex_instance, table):
        query = "what are all the phone numbers?"
        tbl_name = "table"
        table = pd.concat([table] * 500)
        for model in [omnitab_instance, tapas_instance, tapex_instance]:
            model_input = model.process_input(table, query, tbl_name)
            # Assert
            assert model_input is None

    # Test case for successful API response
    def test_predict_input(self, omnitab_instance, tapas_instance, tapex_instance, table):
        query = "what are all the phone numbers?"
        tbl_name = "table"
        for model in [omnitab_instance, tapas_instance, tapex_instance]:
            model_input = model.process_input(table, query, tbl_name)
            result = model.predict_input(model_input, table)
            assert result is not None
