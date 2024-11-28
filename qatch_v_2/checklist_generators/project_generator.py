import random

from .base_generator import BaseGenerator, SingleQA
from ..connectors import ConnectorTable


class ProjectGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'PROJECT'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        columns = list(table.tbl_col2metadata.keys())
        tbl_name = table.tbl_name

        select_tests = []
        select_tests += self._project_single_col(columns, tbl_name)
        select_tests += self._project_all_table(tbl_name)
        select_tests += self._project_add_col(columns, tbl_name)
        select_tests += self._project_random_combination_cols(columns, tbl_name)

        return select_tests

    def _project_all_table(self, tbl_name) -> list[SingleQA]:
        return [SingleQA(
            query=f'SELECT * FROM `{tbl_name}`',
            question=f"Show all the rows in the table {tbl_name}",
            sql_tag='SELECT-ALL',
        )]

    def _project_single_col(self, columns, tbl_name) -> list[SingleQA]:
        output = []
        for col_name in columns:
            test = SingleQA(
                query=f'SELECT `{col_name}` FROM `{tbl_name}`',
                question=f'Show all {col_name} in the table {tbl_name}',
                sql_tag='SELECT-SINGLE-COL',
            )
            output.append(test)
        return output

    def _project_add_col(self, columns, tbl_name) -> list[SingleQA]:
        output = []
        for i in range(1, len(columns)):
            selected_cols = columns[:i]
            query_cols = ", ".join([f'`{col}`' for col in selected_cols])
            question_cols = ", ".join([col for col in selected_cols])
            test = SingleQA(
                query=f'SELECT {query_cols} FROM `{tbl_name}`',
                question=f'Show all {question_cols} in the table {tbl_name}',
                sql_tag='SELECT-ADD-COL',
            )
            output.append(test)
        return output

    def _project_random_combination_cols(self, columns, tbl_name) -> list[SingleQA]:
        output = []
        for i in range(1, len(columns)):
            random_columns = random.sample(columns, i)
            query_cols = ", ".join([f'`{col}`' for col in random_columns])
            question_cols = ", ".join([col for col in random_columns])
            test = SingleQA(
                query=f'SELECT {query_cols} FROM `{tbl_name}`',
                question=f'Show all {question_cols} in the table {tbl_name}',
                sql_tag='SELECT-RANDOM-COL',
            )
            output.append(test)
        return output
