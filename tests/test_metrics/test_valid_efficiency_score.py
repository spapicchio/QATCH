import os.path

import pandas as pd
import pytest

from qatch.connectors import SqliteConnector
from qatch.evaluate_dataset.metrics_evaluators.valid_efficiency_score import ValidEfficiencyScore


class TestValidEfficiencyScore:
    @pytest.fixture
    def instance(self):
        return ValidEfficiencyScore()

    @pytest.fixture
    def connector(self, tmp_path):
        data = {
            "id": [0, 1, 2, 3, 4, 5],
            "year": [1896, 1900, 1904, 2004, 2008, 2012],
            "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
        }
        table = pd.DataFrame.from_dict(data)
        db_tables = {'olympic_games': table}
        connector = SqliteConnector(
            relative_db_path=os.path.join(tmp_path, 'temp.sqlite'),
            db_name='olympic_games',
            tables=db_tables,
            table2primary_key=None
        )
        return connector

    def test_correct_prediction(self, instance, connector):
        target = prediction = 'SELECT * FROM olympic_games'

        ves = instance.run_metric(
            predicted_query=prediction,
            target_query=target,
            target=connector.run_query(target),
            prediction=connector.run_query(prediction),
            connector=connector
        )
        # should not be None
        assert ves

    def test_wrong_prediction(self, instance, connector):
        target = 'SELECT * FROM olympic_games'
        prediction = 'SELECT id from olympic_games WHERE year = 1896'

        ves = instance.run_metric(
            predicted_query=prediction,
            target_query=target,
            target=connector.run_query(target),
            prediction=connector.run_query(prediction),
            connector=connector
        )
        # should not be None
        assert ves == 0
