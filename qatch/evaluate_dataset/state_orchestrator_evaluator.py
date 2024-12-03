import operator
from typing import TypedDict

from pydantic import BaseModel
from typing_extensions import Annotated

from ..connectors import BaseConnector


class PredictedTest(BaseModel):
    target_query: str
    target_values: list[list]
    predicted_query: str
    predicted_values: list[list]


class StateOrchestratorEvaluator(TypedDict):
    connector: BaseConnector
    predicted_test: PredictedTest
    evaluated_tests: Annotated[list, operator.add]
