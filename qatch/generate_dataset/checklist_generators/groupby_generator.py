from qatch.connectors import ConnectorTable
from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample


class GrouByGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'GROUPBY'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Method to generate a list of SingleQA objects based on the category and numeric columns of a given table.

        it generates a list of SingleQA objects by calling two methods - 'generate_group_count_cat'
        using the category columns and 'generate_group_cat_agg_num' using both category and numeric columns.

        Args:
            table (ConnectorTable): The table object containing columns metadata.

        Returns:
            list[SingleQA]: A list of SingleQA objects.

        Note:
            Total number of maximum generated tests: 21

        """
        cat_cols = list(table.cat_col2metadata.keys())
        num_cols = list(table.num_col2metadata.keys())
        table_name = table.tbl_name
        tests = []
        tests += self.generate_group_count_cat(cat_cols, table_name)
        tests += self.generate_group_cat_agg_num(cat_cols, num_cols, table_name)

        return tests

    def generate_group_count_cat(self, cat_columns: list[str], table_name: str) -> list[SingleQA]:
        """
        This function generates a list of test cases, each of which consists of a SQL query and a question based on it.
        The SQL query does a GROUP BY operation on a list of categorical columns and counts the occurrence of each category
        in the said column. The corresponding question is a natural language equivalent of the SQL operation.


        Args:
            cat_columns (list[str]): List of categorical columns on which the operation is performed.
            table_name (str): The name of the table where the operation shall be performed.

        Returns:
            list[SingleQA]: A list of SingleQA objects. Each SingleQA consists of a SQL query, a corresponding
                             question in natural language, and an associated SQL tag marking the type of operation.

        Note:
            - Categorical column sampling limit is up to 5.
            - Max num of tests it can generate: len(cat_columns) = 5
        """

        # num of tests len(cat_col)
        tests = []
        cat_columns = utils_list_sample(cat_columns, k=5, val=self.column_to_include)
        for cat_col in cat_columns:
            single_test = SingleQA(
                query=f'SELECT `{cat_col}`, COUNT(*) FROM `{table_name}` GROUP BY `{cat_col}`',
                question=f'For each {cat_col}, count the number of rows in table {table_name}',
                sql_tag=f'GROUPBY-COUNT',
            )
            tests.append(single_test)
        return tests

    def generate_group_cat_agg_num(self, cat_cols: list[str], num_cols: list[str], table_name: str) -> list[SingleQA]:
        """
        This function generates a list of 'SingleQA' instances that represent several SQL queries and their corresponding
        English-language questions. The SQL queries are generated based on the given categorical and numerical column names
        and the specified operations. Those operations are applied in a GROUP BY context,
        (1) grouping by the categorical column and (2) performing the operation on the numerical column.

        Note:
        - This function excludes numerical columns that contain 'id' in their name.
        - It samples 2 categorical and numerical columns for the generation process to avoid explosion.
        - The resulting queries and questions pertain to finding the min, max, avg, and sum of the numerical column
        for each category in the categorical column.
        - Max number of tests:  len(cat_cols) x len(num_cols) x len(operations) = 2 x 2 x 4 = 16

        Args:
            cat_cols (List[str]): A list of categorical columns names.
            num_cols (List[str]): A list of numerical columns names.
            table_name (str): Name of the table in the database.

        Returns:
            List[SingleQA]: List of 'SingleQA' instances, each representing a SQL query, its corresponding English
            question, and a tag for the type of query.
        """

        # num tests = len(cat_cols) x len(num_cols) x len(operations)
        operations = [
            'min',
            'max',
            'avg',
            'sum',
        ]
        tests = []

        num_cols = [col for col in num_cols if 'id' not in col.lower()]

        cat_cols = utils_list_sample(cat_cols, k=2, val=self.column_to_include)
        num_cols = utils_list_sample(num_cols, k=2, val=self.column_to_include)

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
