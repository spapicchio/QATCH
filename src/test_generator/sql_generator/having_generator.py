from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class HavingGenerator(AbstractSqlGenerator):

    def __init__(self, database: SingleDatabase, seed=2023):
        super().__init__(database, seed)
        self.operation2str = {'>': 'is greater than',
                              '<': 'is less than',
                              '>=': 'is greater than or equal to',
                              '<=': 'is less than or equal to'}
        self.sql_generated = {'sql_tags': [], 'queries': [], 'questions': [], 'results': []}

    def sql_generate(self, table_name: str) -> dict[str, list]:
        self.sql_generated = {'sql_tags': [], 'queries': [], 'questions': [], 'results': []}
        self._build_having_count(table_name)
        self._build_having_agg(table_name)
        return self.sql_generated

    def _build_having_count(self, table_name):
        """
        SELECT policy_type_code FROM policies GROUP BY policy_type_code HAVING count(*)>2
        Find all the policy types that have more than 2 records
        """
        df, cat_cols, _ = self._get_df_cat_num_cols(table_name)
        for cat_col in cat_cols:
            mean_count = int(df.groupby(cat_col).count().mean().values[0])
            queries = [
                f'SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) >= {mean_count}',
                f'SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) <= {mean_count}',
                f'SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) = {mean_count}'
            ]

            questions = [
                f'Find all the "{cat_col}" that have at least {mean_count} records in table "{table_name}"',
                f'Find all the "{cat_col}" that have at most {mean_count} records in table "{table_name}"',
                f'Find all the "{cat_col}" that have exactly {mean_count} records in table "{table_name}"'
            ]

            sql_tags = ['HAVING-COUNT-GR', 'HAVING-COUNT-LS', 'HAVING-COUNT-EQ']

            results = [self.database.run_query(query) for query in queries]

            self.extend_values_generated(sql_tags, queries, questions, results)

    def _build_having_agg(self, table_name):
        """
        SELECT Product_Name FROM PRODUCTS GROUP BY Product_Name HAVING avg(Product_Price) < 1000000
        Find the product names whose average product price is below 1000000.
        """
        # with sample == 2 we get 4 tests for each aggregation -> 4*4 = 16 tests
        # with sample == 3 we get 9 tests for each aggregation -> 9*4 = 36 tests
        df, cat_cols, num_cols = self._get_df_cat_num_cols(table_name, sample=2)
        for cat_col in cat_cols:
            # the mean for each grouped category
            mean_sum = df.groupby(cat_col).sum(numeric_only=True)
            mean_mean = df.groupby(cat_col).mean(numeric_only=True)
            for num_col in num_cols:
                # the mean of sum for the grouped category
                mean_mean_sum = round(mean_sum[num_col].mean(), 2)
                mean_mean_mean = round(mean_mean[num_col].mean(), 2)
                queries = [
                    f'SELECT "{cat_col}" FROM "{table_name}"'
                    f' GROUP BY "{cat_col}" HAVING AVG("{num_col}") >= {mean_mean_mean}',
                    f'SELECT "{cat_col}" FROM "{table_name}"'
                    f' GROUP BY "{cat_col}" HAVING AVG("{num_col}") <= {mean_mean_mean}',

                    f'SELECT "{cat_col}" FROM "{table_name}"'
                    f' GROUP BY "{cat_col}" HAVING SUM("{num_col}") >= {mean_mean_sum}',
                    f'SELECT "{cat_col}" FROM "{table_name}"'
                    f' GROUP BY "{cat_col}" HAVING SUM("{num_col}") <= {mean_mean_sum}',
                ]

                questions = [
                    f'List the "{cat_col}" which average "{num_col}" is at least {mean_mean_mean} in table "{table_name}"',
                    f'List the "{cat_col}" which average "{num_col}" is at most {mean_mean_mean} in table "{table_name}"',

                    f'List the "{cat_col}" which summation of "{num_col}" is at least {mean_mean_sum} in table "{table_name}"',
                    f'List the "{cat_col}" which summation of "{num_col}" is at most {mean_mean_sum} in table "{table_name}"',
                ]

                sql_tags = ['HAVING-AGG-AVG-GR', 'HAVING-AGG-AVG-LS',
                            'HAVING-AGG-SUM-GR', 'HAVING-AGG-SUM-LS']

                results = [self.database.run_query(query) for query in queries]

                self.extend_values_generated(sql_tags, queries, questions, results)

    def extend_values_generated(self, sql_tags, queries, questions, results):
        self.sql_generated['sql_tags'].extend(sql_tags)
        self.sql_generated['queries'].extend(queries)
        self.sql_generated['questions'].extend(questions)
        self.sql_generated['results'].extend(results)
