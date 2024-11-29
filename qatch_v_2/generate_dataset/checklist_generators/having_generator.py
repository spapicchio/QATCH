from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample
from qatch_v_2.connectors import ConnectorTable


class HavingGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'HAVING'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        table_name = table.tbl_name
        tests = []
        cat_cols = table.cat_col2metadata.keys()
        num_cols = table.num_col2metadata.keys()
        tests += self.test_having_count_cat(cat_cols, table_name)
        tests += self.test_having_agg_num(cat_cols, num_cols, table_name)
        return tests

    def test_having_count_cat(self, cat_cols, table_name) -> list[SingleQA]:
        # num tests = len(cat_cols) x len(operations)
        cat_cols = utils_list_sample(cat_cols, k=3)

        operations = [
            ('>=', 'at least'),
            ('<=', 'at most'),
        ]
        tests = []

        for cat_col in cat_cols:
            average_count = self._get_average_of_count_cat_col(table_name, cat_col)
            for op in operations:
                single_test = SingleQA(
                    query=f"SELECT `{cat_col}` FROM `{table_name}` GROUP BY `{cat_col}`"
                          f" HAVING count(*) {op[0]} {average_count}",
                    question=f'Find all the {cat_col} that have {op[1]} {average_count} records in table {table_name}',
                    sql_tag=f'HAVING-COUNT',
                )
                tests.append(single_test)
        return tests

    def test_having_agg_num(self, cat_cols, num_cols, table_name):
        # num tests = len(cat_cols) x len(num_cols) x len(operations) x len(symbols)
        cat_cols = utils_list_sample(cat_cols, k=2)
        num_cols = utils_list_sample(num_cols, k=2)

        tests = []
        operations = [
            ('AVG', 'average'),
            ('SUM', 'summation'),
        ]
        symbols = [
            (">=", "at least"),
            ("<=", "at most"),
        ]

        # remove num_cols with ID. No sense to calculate sum/avg over ids
        num_cols = [col for col in num_cols if 'id' not in col.lower()]

        for cat_col in cat_cols:
            for num_col in num_cols:
                mean_mean_sum, mean_mean_mean = self._get_average_of_sum_avg_cat_col(table_name, cat_col, num_col)
                for op in operations:
                    for symbol in symbols:
                        value = mean_mean_mean if op[0] == 'AVG' else mean_mean_sum
                        single_test = SingleQA(
                            query=f'SELECT `{cat_col}` FROM `{table_name}`'
                                  f' GROUP BY `{cat_col}` HAVING {op[0]}(`{num_col}`) {symbol[0]} {value}',
                            question=f'List the {cat_col} which {op[1]} of {num_col} '
                                     f'is {symbol[1]} {value} in table {table_name}',
                            sql_tag=f'HAVING-AGG-{op[0]}',
                        )
                        tests.append(single_test)
        return tests

    def _get_average_of_count_cat_col(self, table_name, cat_col):
        """
        Helper method to calculate the average count of rows for each category in a categorical column.

        Args:
            table_name (str): The name of the table in the database.
            cat_col (str): The name of the categorical column.

        Returns:
            int: The average count of rows for each category.
        """
        # SQL query to get the average count for each category
        inner_query = f'SELECT COUNT(*) AS row_count FROM `{table_name}` GROUP BY `{cat_col}`'
        # Run the inner query and get the average of row counts
        average = self.connector.run_query(f'SELECT AVG(row_count) FROM ({inner_query})')[0][0]
        return int(average)

    def _get_average_of_sum_avg_cat_col(self, table_name, cat_col, num_col):
        """
        Helper method to calculate the average sum and average of a numerical column for each category in a categorical column.

        Args:
            table_name (str): The name of the table in the database.
            cat_col (str): The name of the categorical column.
            num_col (str): The name of the numerical column.

        Returns:
            tuple: A tuple containing the average sum and average of the numerical column for each category.
        """
        # SQL queries to get the average sum and average of numerical column for each category
        inner_query_sum = f'SELECT SUM(`{num_col}`) AS sum_col FROM `{table_name}` GROUP BY `{cat_col}`'
        inner_query_avg = f'SELECT AVG(`{num_col}`) AS avg_col FROM `{table_name}` GROUP BY `{cat_col}`'
        # Run the inner queries and get the average of sums and averages
        average_sum = self.connector.run_query(f'SELECT AVG(sum_col) FROM ({inner_query_sum})')[0][0]
        average_avg = self.connector.run_query(f'SELECT AVG(avg_col) FROM ({inner_query_avg})')[0][0]
        return round(average_sum, 2), round(average_avg, 2)
