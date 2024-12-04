from qatch.connectors import ConnectorTable
from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample


class SimpleAggGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'SIMPLE-AGG'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Generates a list of SingleQA dictionaries structured for testing aggregator methods.
        Uses two helper methods, `generate_count_cat` and `generate_agg_num`,
        to parse categorical and numerical metadata
        from the `ConnectorTable` object.

        Args:
            table (ConnectorTable): A ConnectorTable object, which includes attributes like table name,
                                    categorical and numerical column metadata, etc.

        Returns:
            list[SingleQA]: Returns a list of SingleQA dictionaries structured for testing each column in the table.
                            Each SingleQA dictionary includes keys like "query", "question", and "sql_tag".
        """

        cat_columns = list(table.cat_col2metadata.keys())
        num_cols = list(table.num_col2metadata.keys())
        table_name = table.tbl_name
        tests = self.generate_count_cat(cat_columns, table_name)
        tests += self.generate_agg_num(num_cols, table_name)

        return tests

    def generate_count_cat(self, cat_columns: list[str], table_name: str) -> list[SingleQA]:
        """
        Generates a list of SingleQA tests that count the distinct categorical values in each specified column
        of a given table.

        This method takes a list of categorical column names and a table name, then formulates SQL queries to
        count the distinct values in each of those columns. It then packages these SQL queries into SingleQA tests.

        Args:
            cat_columns (list[str]): The list of names of categorical columns in the table.
            table_name (str): The name of the table in the database.

        Returns:
            list[SingleQA]: A list of SingleQA tests.

        Note:
            Only the first 5 selected categorical columns will be used for generating SQL queries to avoid explosion.
        """

        # num tests = len(cat_columns)
        cat_columns = utils_list_sample(cat_columns, k=5, val=self.column_to_include)

        tests = []
        for cat_col in cat_columns:
            single_test = SingleQA(
                query=f'SELECT COUNT(DISTINCT `{cat_col}`) FROM `{table_name}`',
                question=f'How many different {cat_col} are in table {table_name}?',
                sql_tag='SIMPLE-AGG-COUNT-DISTINCT',
            )
            tests.append(single_test)
        return tests

    def generate_agg_num(self, num_cols: list[str], table_name: str) -> list[SingleQA]:
        """
        Generates a list of SingleQA tests based on provided numerical columns and a table name.

        This function creates tests using aggregation operations such as MAX, MIX, and AVG.
        It doesn't include columns with 'ID' as they are usually not meaningful for these operations.

        Args:
            num_cols (list[str]): List of numerical column names from which
                                  to generate the tests.
            table_name (str): The name of the table where the columns reside.

        Returns:
            list[SingleQA]: A list of SingleQA test instances, where each test includes a SQL query,
                            a corresponding natural language question, and a SQL tag.

        Note:
            - The number of tests generated equals len(num_cols) x len(operations)
            where operations are ['MAX', 'MIX', 'AVG']. Columns with 'ID' in their names are ignored.
            - To avoid explosion, only two numerical columns are sampled and used in the generation.
        """

        # num tests = len(num_cols) x len(operations)

        # remove num_cols with ID. No meaning to calculate max/min/avg over ids
        num_cols = [col for col in num_cols if 'id' not in col.lower()]
        num_cols = utils_list_sample(num_cols, k=2, val=self.column_to_include)

        operations = [
            ('MAX', 'maximum'),
            ('MIN', 'minimum'),
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
