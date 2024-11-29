from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample
from qatch_v_2.connectors import ConnectorTable


class SimpleAggGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'SIMPLE-AGG'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        cat_columns = table.cat_col2metadata.keys()
        num_cols = table.num_col2metadata.keys()
        table_name = table.tbl_name
        tests = []
        tests += self.test_count_cat(cat_columns, table_name)
        tests += self.test_agg_num(num_cols, table_name)

        return tests

    def test_count_cat(self, cat_columns, table_name):
        # num tests = len(cat_columns)
        cat_columns = utils_list_sample(cat_columns, k=5)

        tests = []
        for cat_col in cat_columns:
            single_test = SingleQA(
                query=f'SELECT COUNT(DISTINCT `{cat_col}`) FROM `{table_name}`',
                question=f'How many different {cat_col} are in table {table_name}?',
                sql_tag='SIMPLE-AGG-COUNT-DISTINCT',
            )
            tests.append(single_test)
        return tests

    def test_agg_num(self, num_cols, table_name):
        # num tests = len(num_cols) x len(operations)

        # remove num_cols with ID. No meaning to calculate max/min/avg over ids
        num_cols = [col for col in num_cols if 'id' not in col.lower()]
        num_cols = utils_list_sample(num_cols, k=2)

        operations = [
            ('MAX', 'maximum'),
            ('MIX', 'minimum'),
            ('AVG', 'average'),
        ]
        tests = []
        for num_col in num_cols:
            for operation in operations:
                single_test = SingleQA(
                    query=f'SELECT {operation[0]}(`{num_col}`) FROM `{table_name}`',
                    question=f'Find the {operation[1]} {num_col} for the table {table_name}',
                    sql_tag=f'SIMPLE-AGG-{operation[0]}',
                )
                tests.append(single_test)
        return tests
