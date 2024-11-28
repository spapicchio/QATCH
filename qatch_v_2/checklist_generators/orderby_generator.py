from .base_generator import BaseGenerator, SingleQA
from ..connectors import ConnectorTable


class OrderByGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'ORDERBY'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        columns = list(table.tbl_col2metadata.keys())
        tbl_name = table.tbl_name

        select_tests = []
        select_tests += self._all_table_order(columns, tbl_name)
        select_tests += self._single_col_order(columns, tbl_name)

        return select_tests

    def _all_table_order(self, columns, tbl_name) -> list[SingleQA]:
        tests = []
        operations = [
            ('ASC', 'ascending'),
            ('DESC', 'descending'),
        ]
        for col in columns:
            for operation in operations:
                single_qa = SingleQA(
                    query=f'SELECT * FROM `{tbl_name}` ORDER BY `{col}` {operation[0]}',
                    question=f'Show all data ordered by {col} in {operation[1]} order for the table {tbl_name}',
                    sql_tag='ORDERBY-SINGLE',
                )
                tests.append(single_qa)
        return tests

    def _single_col_order(self, columns, tbl_name) -> list[SingleQA]:
        tests = []
        operations = [
            ('ASC', 'ascending'),
            ('DESC', 'descending'),
        ]
        for col in columns:
            for operation in operations:
                single_qa = SingleQA(
                    query=f'SELECT `{col}` FROM `{tbl_name}` ORDER BY `{col}` {operation[0]}',
                    question=f'Project the {col} ordered in {operation[1]} order for the table {tbl_name}',
                    sql_tag='ORDERBY-PROJECT',
                )
                tests.append(single_qa)
        return tests
