from __future__ import annotations

import logging

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
    JoinGenerator,
    ManyToManyGenerator

)
from .state_orchestrator_generator import StateOrchestratorGenerator
from ..connectors import Connector

name2generator = {
    'project': ProjectGenerator,
    'distinct': DistinctGenerator,
    'select': SelectGenerator,
    'simple': SimpleAggGenerator,
    'orderby': OrderByGenerator,
    'groupby': GrouByGenerator,
    'having': HavingGenerator,
    'join': JoinGenerator,
    'many-to-many': ManyToManyGenerator,
}


class OrchestratorGenerator:
    def __init__(self, generator_names: list[str] | None = None):
        graph = StateGraph(StateOrchestratorGenerator)

        if generator_names is None:
            generator_names = name2generator.keys()

        list_node_fun = [
            (name, name2generator[name]().graph_call)
            for name in generator_names
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
        if len(dataset) > 0:
            dataset = dataset.loc[:, ['db_path', 'db_id', 'tbl_name', 'test_category', 'sql_tag', 'query', 'question']]
        else:
            logging.warning(f'QATCH not able to generate tests from {connector.db_path}')
        return dataset
