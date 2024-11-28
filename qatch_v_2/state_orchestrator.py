import operator
from typing import Annotated, TypedDict

from .connectors import Connector


class StateOrchestrator(TypedDict):
    connector: Connector
    generated_templates: Annotated[list, operator.add]
