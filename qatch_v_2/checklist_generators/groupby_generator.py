from .base_generator import BaseGenerator, SingleQA
from ..connectors import ConnectorTable


class GrouByGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'GROUPBY'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        cat_cols = table.cat_col2metadata.keys()
        num_cols = table.num_col2metadata.keys()
        table_name = table.tbl_name
        tests = []
        tests += self.test_group_count_cat(cat_cols, table_name)
        tests += self.test_group_cat_agg_num(cat_cols, num_cols, table_name)

        return tests

    def test_group_count_cat(self, cat_columns, table_name):
        tests = []
        for cat_col in cat_columns:
            single_test = SingleQA(
                query=f'SELECT `{cat_col}`, COUNT(*) FROM `{table_name}` GROUP BY `{cat_col}`',
                question=f'For each {cat_col}, count the number of rows in table {table_name}',
                sql_tag=f'GROUPBY-COUNT',
            )
            tests.append(single_test)
        return tests

    def test_group_cat_agg_num(self, cat_cols, num_cols, table_name):
        operations = [
            'min',
            'max',
            'avg',
            'sum',
        ]
        tests = []

        for cat_col in cat_cols:
            for num_col in num_cols:
                for operation in operations:
                    single_test = SingleQA(
                        query=f'SELECT `{cat_col}`, {operation.upper()}(`{num_col}`) '
                              f'FROM `{table_name}` GROUP BY `{cat_col}`',
                        question=f'For each {cat_col}, find the {operation} of {num_col} in table {table_name}',
                        sql_tag=f'GROUPBY-AGG-{operation.upper()}',
                    )
                    tests.append(single_test)

        return tests
