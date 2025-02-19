from __future__ import annotations

import operator
from typing import TypedDict

from pydantic import BaseModel
from typing_extensions import Annotated

from ..connectors import BaseConnector


class PredictedTest(BaseModel):
    target_query: str | list[list]
    target_values: list[list]
    predicted_query: str | list[list]
    predicted_values: list[list]


class StateOrchestratorEvaluator(TypedDict):
    connector: BaseConnector
    predicted_test: PredictedTest
    evaluated_tests: Annotated[list, operator.add]
