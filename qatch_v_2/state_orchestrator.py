import operator
from typing import Annotated, TypedDict

from .connectors import ConnectorTable


class StateOrchestrator(TypedDict):
    connector: ConnectorTable
    database: dict[str, ConnectorTable]
    generated_templates: Annotated[list, operator.add]
