import random
from abc import ABC, abstractmethod
from itertools import chain
from typing import TypedDict, Literal

from qatch_v_2.connectors import ConnectorTable
from ..state_orchestrator_generator import StateOrchestratorGenerator


class SingleQA(TypedDict):
    query: str
    question: str
    sql_tag: str


class BaseTest(SingleQA):
    db_id: str
    db_path: str
    tbl_name: str
    test_category: str


class BaseGenerator(ABC):
    def __init__(self, seed=2023):
        random.seed(seed)
        self.connector = None

    @property
    @abstractmethod
    def test_name(self):
        raise NotImplementedError

    def graph_call(self, state: StateOrchestratorGenerator) -> dict[Literal['generated_templates']: list[BaseTest]]:
        database = state['database']
        connector = self.connector = state['connector']
        table_tests = []
        for tbl_name, table in database.items():
            tests = self.template_generator(table)

            tests = [self.create_base_test(table, test) for test in tests]

            table_tests.append(tests)
        # flatten the table tests
        table_tests = list(chain.from_iterable(table_tests))

        # remove empty tests
        table_tests = self.remove_test_with_empty_results(table_tests, connector)

        return {'generated_templates': table_tests}

    def create_base_test(self, table, test: SingleQA) -> BaseTest:
        return BaseTest(
            **test,
            db_id=table.db_name,
            db_path=table.db_path,
            tbl_name=table.tbl_name,
            test_category=self.test_name
        )

    def remove_test_with_empty_results(self, tests: list[SingleQA], connector) -> list[SingleQA]:
        return [test for test in tests if connector.run_query(test['query']) != []]

    @abstractmethod
    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        raise NotImplementedError