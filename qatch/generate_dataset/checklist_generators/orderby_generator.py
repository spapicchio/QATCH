from qatch.connectors import ConnectorTable
from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample


class OrderByGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'ORDERBY'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        columns = list(table.tbl_col2metadata.keys())
        tbl_name = table.tbl_name

        select_tests = self.generate_all_table_order(columns, tbl_name)
        select_tests += self.generate_single_col_order(columns, tbl_name)

        return select_tests

    def generate_all_table_order(self, columns, tbl_name) -> list[SingleQA]:
        """
        This method generates SQL queries and respective questions for testing
        ORDER BY clause on every column of a table, in both ascending and
        descending order. It returns a list of SingleQA typed dictionaries,
        where each SingleQA dictionary contains an SQL query, its respective
        question and a test tag.

        Args:
            columns (List[str]): List of column names of the table.
            tbl_name (str): Name of the table.

        Returns:
            List[SingleQA]: A list of SingleQA typed dictionaries. Each
            SingleQA contains:
                query (str): The generated SQL query.
                question (str): The corresponding question of the SQL query.
                sql_tag (str): Tag representing the type of SQL query.

        Notes:
            - This method always uses a sample of only two columns from the
            provided list, even if more columns are provided to avoid explosion.
        """

        # number of tests: len(columns) * 2
        columns = utils_list_sample(columns, k=2, val=self.column_to_include)

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

    def generate_single_col_order(self, columns, tbl_name) -> list[SingleQA]:
        """
        Creates a list of `SingleQA` objects representing queries that select a random column from
        the provided columns and sort the selected column data in ascending or descending order.

        The function takes a list of columns and randomly selects one column using the `utils_list_sample` function.
        For the selected column, it creates queries to sort the data in both ascending (ASC) and descending (DESC) order.

        Notes:
            This function generates `len(columns) * 2` number of tests, as for each column two tests
            (ascending and descending order) are created. The column used for generating the tests is
            selected randomly from the provided list of columns.

        Args:
            columns (list): A list of column names for which the ascending and descending sort queries are to be created.
            tbl_name (str): The name of the table that contains the columns for which the queries will be generated.

        Returns:
            list[SingleQA]: A list of `SingleQA` objects representing the tests. Each `SingleQA` object contains
                             the query, the question to be asked i.e., the projection of the column in a particular
                             (ascending or descending) order, and the SQL tag for the test.

        """

        # number of tests: len(columns) * 2
        columns = utils_list_sample(columns, k=2, val=self.column_to_include)

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
