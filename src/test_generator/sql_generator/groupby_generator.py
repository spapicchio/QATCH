from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class GroupByGenerator(AbstractSqlGenerator):
    def __init__(self, database: SingleDatabase, seed=2023):
        super().__init__(database, seed)
        self.sql_generated = {'sql_tags': [], 'queries': [], 'questions': [], 'results': []}

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """the group by is performed only with the categorical columns"""
        self.sql_generated = {'sql_tags': [], 'queries': [], 'questions': [], 'results': []}
        df = self.database.tables[table_name]
        self._build_group_by_no_agg(table_name)
        self._build_group_by_with_count(table_name)
        self._build_group_by_with_agg(table_name)
        return self.sql_generated

    def _build_group_by_no_agg(self, table_name):
        cat_cols, _ = self._get_cat_num_cols(table_name)
        random_combinations = self._comb_random(cat_cols)

        questions = [f'Show all {self._get_col_comb_str(comb)}' \
                     f' in the table "{table_name}" for each {self._get_col_comb_str(comb)}'
                     for comb in random_combinations]

        queries = [f'SELECT {self._get_col_comb_str(comb)} FROM ' \
                   f'"{table_name}" GROUP BY {self._get_col_comb_str(comb)}'
                   for comb in random_combinations]

        results = [self.database.run_query(query) for query in queries]

        sql_tags = ['GROUPBY-NO-AGGR'] * len(queries)

        self.extend_values_generated(sql_tags=sql_tags, queries=queries,
                                     questions=questions, results=results)

    def _build_group_by_with_count(self, table_name):
        """only for Categorcical columns"""
        cat_cols, _ = self._get_cat_num_cols(table_name)
        questions = [f'For each "{col}", count the number of rows in table "{table_name}"'
                     for col in cat_cols]
        queries = [f'SELECT "{col}", COUNT(*) FROM "{table_name}" GROUP BY "{col}"'
                   for col in cat_cols]
        results = [self.database.run_query(query) for query in queries]
        sql_tags = ['GROUPBY-COUNT'] * len(queries)

        self.extend_values_generated(sql_tags=sql_tags, queries=queries,
                                     questions=questions, results=results)

    def _build_group_by_with_agg(self, table_name):
        """only for Numerical columns"""
        cat_cols, num_cols = self._get_cat_num_cols(table_name)
        for agg in ['min', 'max', 'avg', 'sum']:
            questions = [f'For each "{c_col}", find the {agg} of "{n_col}" in table "{table_name}"'
                         for c_col in cat_cols
                         for n_col in num_cols]

            queries = [f'SELECT "{c_col}", {agg.upper()}("{n_col}") FROM "{table_name}" GROUP BY {c_col}'
                       for c_col in cat_cols
                       for n_col in num_cols]

            results = [self.database.run_query(query) for query in queries]

            sql_tags = [f'GROUPBY-AGG-{agg.upper()}'] * len(results)

            self.extend_values_generated(sql_tags=sql_tags, queries=queries,
                                         questions=questions, results=results)

    def extend_values_generated(self, sql_tags, queries, questions, results):
        self.sql_generated['sql_tags'].extend(sql_tags)
        self.sql_generated['queries'].extend(queries)
        self.sql_generated['questions'].extend(questions)
        self.sql_generated['results'].extend(results)
