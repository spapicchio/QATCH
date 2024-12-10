from qatch.connectors import ConnectorTable
from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample


class DistinctGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'DISTINCT'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Generates SingleQA templates using provided table's categorical columns.
        A SingleQA is a dictionary having query, question and SQL tag.
        The main purpose is to test distinct values of the columns.

        Generate tests which are based on querying distinct values from a few
        (up to 5) categorical columns from the input table.


        Args:
            table (ConnectorTable): The input table to generate tests from.

        Returns:
            list[SingleQA]: List of unique SQL queries to test distinct values in columns.

        Note:
            - Categorical column sampling limit is up to 5.
            - Max num of tests it can generate: len(cat_columns) < 5
        """
        # num of tests len(cat_columns)
        cat_columns = list(table.cat_col2metadata.keys())
        cat_columns = utils_list_sample(cat_columns, k=5, val=self.column_to_include)

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
