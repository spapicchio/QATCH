import operator
from typing import TypedDict

from pydantic import BaseModel
from typing_extensions import Annotated


class PredictedTest(BaseModel):
    target: list[list]
    prediction: list[list]


class StateOrchestratorEvaluator(TypedDict):
    predicted_test: PredictedTest
    evaluated_tests: Annotated[list, operator.add]
