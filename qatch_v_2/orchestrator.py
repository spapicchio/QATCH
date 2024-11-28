from __future__ import annotations

from typing import Callable

import pandas as pd
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from .checklist_generators.select_generator import SelectGenerator
from .connectors import Connector
from .state_orchestrator import StateOrchestrator


class Orchestrator:
    def __init__(self, list_node_fun: list[tuple[str, Callable]] | None = None):
        graph = StateGraph(StateOrchestrator)
        if list_node_fun is None:
            list_node_fun = [

                ('Project', SelectGenerator().graph_call),
            ]
        for node_name, node_fun in list_node_fun:
            graph.add_node(node_name, node_fun)
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        self.graph = graph.compile()

    def generate_dataset(self, connector: Connector) -> pd.DataFrame:
        state = self.graph.invoke({'connector': connector})
        dataset = state['generated_templates']
        dataset = pd.DataFrame(dataset)
        dataset = dataset.loc[:, ['db_path', 'db_id', 'tbl_name', 'test_category', 'sql_tag', 'query', 'question']]
        return dataset
