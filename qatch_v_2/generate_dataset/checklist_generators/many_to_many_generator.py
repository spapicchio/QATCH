from .base_generator import BaseGenerator, SingleQA
from ...connectors import ConnectorTable


class ManyToManyGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'many-to-many-generator'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Generates a list of SQL query tests for each pair of categorical columns in a given table.

        The function generates a SQL query for each pair of categorical columns in the table.
        The query checks if for each unique value in one column, there are all unique values of the other column.

        Args:
            table (ConnectorTable): The table instance with data for query generation.

        Returns:
            list[SingleQA]: A list of SQL queries-questions as instances of SingleQA class.

        Note:
            The number of tests (queries) generated is n*(n-1)/2, where n is the number of categorical columns in the table.
            If a table does not have at least two categorical columns, the function will return an empty list.
        """

        # given n = len(cat_cols)
        # len of tests: n*(n-1)/2.
        cat_cols = list(table.cat_col2metadata.keys())
        tests = []

        for i in range(len(cat_cols) - 1):
            for j in range(i + 1, len(cat_cols)):
                selected_cols = [cat_cols[i], cat_cols[j]]
                single_test = SingleQA(
                    query=f"SELECT `{selected_cols[0]}` FROM `{table.tbl_name}`"
                          f" GROUP BY `{selected_cols[0]}` HAVING COUNT(DISTINCT `{selected_cols[1]}`) = ("
                          f"SELECT COUNT(DISTINCT `{selected_cols[1]}`) FROM `{table.tbl_name}`"
                          f")",
                    question=f"What are the {selected_cols[0]} with all the {selected_cols[1]} "
                             f"in table {table.tbl_name}?",
                    sql_tag='many-to-many'
                )
                tests.append(single_test)
        return tests
