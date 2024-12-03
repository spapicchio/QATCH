from .base_generator import BaseGenerator, SingleQA
from qatch.connectors import ConnectorTable


class NullGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'NULL'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Generates a list of SingleQA tests for a given table. Each test is a query that counts the number of
        rows where a particular column's values are null or not null. The columns with 'id' in their names are skipped.

        Args:
            table (ConnectorTable): An instance of ConnectorTable for which tests are to be generated.

        Returns:
            list[SingleQA]: A list of SingleQA instances, each representing a query test for the table.

        Note:
            The number of generated tests equals the number of columns with null values in the table, excluding
            columns with 'id' in their names.
            Most of these tests will be empty and will be removed by `BaseGenerator` class.
        """

        # Mostly of this tests will be empty, but the `BaseGenerator` class will remove them
        # number of test = len(column with null values in table)
        columns = table.tbl_col2metadata.keys()
        table_name = table.tbl_name
        tests = []
        operations = [
            ('IS NULL', 'missing'),
            ('IS NOT NULL', 'not missing'),
        ]

        # remove columns with ID. No sense to calculate sum/avg over ids
        columns = [col for col in columns if 'id' not in col.lower()]

        for col in columns:
            for op in operations:
                test = SingleQA(
                    query=f'SELECT COUNT(*) FROM `{table_name}` WHERE `{col}` {op[0]}',
                    question=f'Count the rows where the values of {col} are {op[1]} in table {table_name}',
                    sql_tag='NOT-NULL-COUNT'
                )
                tests.append(test)
        return tests
