from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class SimpleAggGenerator(AbstractSqlGenerator):
    def __init__(self, database: SingleDatabase, seed=2023):
        super().__init__(database, seed)
        self.sql_generated = {'sql_tags': [], 'queries': [], 'questions': [], 'results': []}

    def sql_generate(self, table_name: str) -> dict[str, list]:
        self.sql_generated = {'sql_tags': [], 'queries': [], 'questions': [], 'results': []}
        cat_cols, num_cols = self._get_cat_num_cols(table_name)
        df = self.database.get_table_from_name(table_name)
        self._build_count_cat(table_name, cat_cols)
        self._build_count_agg(table_name, num_cols)

        assert len(self.sql_generated['sql_tags']) == \
               len(self.sql_generated['queries']) == \
               len(self.sql_generated['questions']) == \
               len(self.sql_generated['results'])

        return self.sql_generated

    def _build_count_cat(self, table_name, cat_cols):
        """
        SELECT COUNT(*) FROM table_name
        Count the records in Customers_Cards?

        SELECT count(DISTINCT card_type_code) FROM Customers_Cards
        How many different card types are there?
        """

        queries = [f'SELECT COUNT(*) FROM "{table_name}"']
        questions = [f'Count the records in table "{table_name}"?']
        sql_tags = ['SIMPLE-AGG-COUNT']

        for cat_col in cat_cols:
            queries += [f'SELECT COUNT(DISTINCT"{cat_col}") FROM "{table_name}"']
            questions += [f'How many different "{cat_col}" are in table "{table_name}"?']
            sql_tags += ['SIMPLE-AGG-COUNT-DISTINCT']

        results = [self.database.run_query(query) for query in queries]
        self.extend_values_generated(sql_tags, queries, questions, results)

    def _build_count_agg(self, table_name, num_cols):
        """
        SELECT max(monthly_rental)FROM Student_Addresses
        Find the maximum monthly rental for the table Student_Addresses.
        """
        for num_col in num_cols:
            queries = [
                f'SELECT MAX("{num_col}") FROM "{table_name}"',
                f'SELECT MIN("{num_col}") FROM "{table_name}"',
                f'SELECT AVG("{num_col}") FROM "{table_name}"'
            ]
            questions = [
                f'Find the maximum "{num_col}" for the table "{table_name}"',
                f'Find the minimum "{num_col}" for the table "{table_name}"',
                f'Find the average "{num_col}" for the table "{table_name}"'
            ]
            results = [self.database.run_query(query) for query in queries]
            sql_tags = ['SIMPLE-AGG-MAX', 'SIMPLE-AGG-MIN', 'SIMPLE-AGG-AVG']
            self.extend_values_generated(sql_tags, queries, questions, results)

    def extend_values_generated(self, sql_tags, queries, questions, results):
        self.sql_generated['sql_tags'].extend(sql_tags)
        self.sql_generated['queries'].extend(queries)
        self.sql_generated['questions'].extend(questions)
        self.sql_generated['results'].extend(results)
