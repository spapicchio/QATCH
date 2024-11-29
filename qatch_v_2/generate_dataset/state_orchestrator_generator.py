import operator
from typing import Annotated, TypedDict

from ..connectors import ConnectorTable, Connector


class StateOrchestratorGenerator(TypedDict):
    connector: Connector
    database: dict[str, ConnectorTable]
    generated_templates: Annotated[list, operator.add]
