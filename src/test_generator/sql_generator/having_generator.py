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
        cat_cols, num_cols = self._get_cat_num_cols(table_name)
        df = self.database.get_table_from_name(table_name)
        self._build_having_count(df, table_name, cat_cols)
        self._build_having_agg(df, table_name, cat_cols, num_cols)
        return self.sql_generated

    def _build_having_count(self, df, table_name, cat_cols):
        """
        SELECT policy_type_code FROM policies GROUP BY policy_type_code HAVING count(*)>2
        Find all the policy types that have more than 2 records
        """
        for cat_col in cat_cols:
            mean_count = int(df.groupby(cat_col).count().mean().values[0])
            queries = [
                f'SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) >= {mean_count}',
                f'SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) <= {mean_count}',
                f'SELECT "{cat_col}" FROM "{table_name}" GROUP BY "{cat_col}" HAVING count(*) = {mean_count}'
            ]

            questions = [
                f'Find all the "{cat_col}" that have at least {mean_count} records',
                f'Find all the "{cat_col}" that have at most {mean_count} records',
                f'Find all the "{cat_col}" that have exactly {mean_count} records'
            ]

            sql_tags = ['HAVING-COUNT-GR', 'HAVING-COUNT-LS', 'HAVING-COUNT-EQ']

            results = [self.database.run_query(query) for query in queries]

            self.extend_values_generated(sql_tags, queries, questions, results)

    def _build_having_agg(self, df, table_name, cat_cols, num_cols):
        """
        SELECT Product_Name FROM PRODUCTS GROUP BY Product_Name HAVING avg(Product_Price) < 1000000
        Find the product names whose average product price is below 1000000.
        """
        for cat_col in cat_cols:
            # the mean for each grouped category
            mean_count = df.groupby(cat_col).mean(numeric_only=True)
            for num_col in num_cols:
                # the mean of means for the grouped category
                mean_mean = mean_count[num_col].mean()
                queries = [
                    f'SELECT "{cat_col}" FROM "{table_name}"'
                    f' GROUP BY "{cat_col}" HAVING MIN("{num_col}") >= {mean_mean}',
                    f'SELECT "{cat_col}" FROM "{table_name}"'
                    f' GROUP BY "{cat_col}" HAVING MAX("{num_col}") <= {mean_mean}',
                ]

                questions = [
                    f'Find the "{cat_col}" whose min "{num_col}" is at least {mean_mean}',
                    f'Find the "{cat_col}" whose max "{num_col}" is at most {mean_mean}',
                ]

                sql_tags = ['HAVING-AGG-GR', 'HAVING-AGG-LS']

                results = [self.database.run_query(query) for query in queries]

                self.extend_values_generated(sql_tags, queries, questions, results)

    def extend_values_generated(self, sql_tags, queries, questions, results):
        self.sql_generated['sql_tags'].extend(sql_tags)
        self.sql_generated['queries'].extend(queries)
        self.sql_generated['questions'].extend(questions)
        self.sql_generated['results'].extend(results)
