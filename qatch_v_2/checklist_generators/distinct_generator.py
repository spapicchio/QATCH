from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample
from ..connectors import ConnectorTable


class DistinctGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'DISTINCT'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        # num of tests len(cat_columns)
        cat_columns = table.cat_col2metadata.keys()
        cat_columns = utils_list_sample(cat_columns, k=5)

        table_name = table.tbl_name
        tests = []
        for cat_col in cat_columns:
            single_test = SingleQA(
                query=f'SELECT DISTINCT `{cat_col}` FROM `{table_name}`',
                question=f'Show the different {cat_col} in the table {table_name}',
                sql_tag='DISTINCT-SINGLE',
            )
            tests.append(single_test)
        return tests
