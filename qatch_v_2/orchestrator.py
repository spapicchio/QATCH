from __future__ import annotations

from typing import Callable

import pandas as pd
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from .checklist_generators import (
    ProjectGenerator,
    SelectGenerator,
    DistinctGenerator,
    SimpleAggGenerator,
    OrderByGenerator,
    GrouByGenerator,
    HavingGenerator,
    JoinGenerator

)
from .connectors import Connector
from .state_orchestrator import StateOrchestrator


class Orchestrator:
    def __init__(self, list_node_fun: list[tuple[str, Callable]] | None = None):
        graph = StateGraph(StateOrchestrator)
        if list_node_fun is None:
            list_node_fun = [

                # ('Project', ProjectGenerator().graph_call),
                # ('Select', SelectGenerator().graph_call),
                # ('Distinct', DistinctGenerator().graph_call),
                # ('SimpleAgg', SimpleAggGenerator().graph_call),
                # ('OrderBy', OrderByGenerator().graph_call),
                # ('Groupby', GrouByGenerator().graph_call),
                # ('Having', HavingGenerator().graph_call),
                ('Join', JoinGenerator().graph_call)
            ]
        for node_name, node_fun in list_node_fun:
            graph.add_node(node_name, node_fun)
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, END)

        self.graph = graph.compile()

    def generate_dataset(self, connector: Connector) -> pd.DataFrame:
        database = connector.load_tables_from_database()
        state = self.graph.invoke({'database': database, 'connector': connector})
        dataset = state['generated_templates']
        dataset = pd.DataFrame(dataset)
        dataset = dataset.loc[:, ['db_path', 'db_id', 'tbl_name', 'test_category', 'sql_tag', 'query', 'question']]
        return dataset
