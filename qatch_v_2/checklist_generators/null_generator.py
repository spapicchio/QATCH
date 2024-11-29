from .base_generator import BaseGenerator, SingleQA
from ..connectors import ConnectorTable


class NullGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'NULL'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        # Mostly of this tests will be empty, but the `BaseGenerator` class will remove them
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
