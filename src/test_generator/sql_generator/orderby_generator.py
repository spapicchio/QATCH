from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class OrderByGenerator(AbstractSqlGenerator):

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str):
        # TODO include order of one column with one select
        self.empty_sql_generated()
        columns = self.database.get_columns_from_table(table_name)
        self.generate_order_asc(table_name, columns)
        self.generate_order_desc(table_name, columns)
        # TODO check when necessary
        # self.generate_order_asc_project(table_name, columns)
        # self.generate_order_desc_project(table_name, columns)
        return self.sql_generated

    def generate_order_asc(self, table_name: str, columns: list[str]):
        queries = [f'SELECT * FROM "{table_name}" ORDER BY "{col}" ASC'
                   for col in columns]

        questions = [
            f'Show all data ordered by "{col}" in ascending order for the table "{table_name}"'
            for col in columns
        ]
        results = [self.database.run_query(query) for query in queries]
        sql_tags = ['ORDERBY-SINGLE'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions, results)
        return self.sql_generated

    def generate_order_desc(self, table_name, columns: list[str]):
        queries = [f'SELECT * FROM "{table_name}" ORDER BY "{col}" DESC'
                   for col in columns]

        questions = [
            f'Show all data ordered by "{col}" in descending order for the table "{table_name}"'
            for col in columns
        ]
        results = [self.database.run_query(query) for query in queries]
        sql_tags = ['ORDERBY-SINGLE'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions, results)
        return self.sql_generated

    def generate_order_asc_project(self, table_name: str, columns: list[str]):
        queries = [f'SELECT "{col}" FROM "{table_name}" ORDER BY "{col}" ASC'
                   for col in columns]

        questions = [
            f'Project the "{col}" ordered in ascending order for the table {table_name}'
            for col in columns
        ]
        results = [self.database.run_query(query) for query in queries]
        sql_tags = ['ORDERBY-PROJECT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions, results)
        return self.sql_generated

    def generate_order_desc_project(self, table_name, columns: list[str]):
        queries = [f'SELECT "{col}" FROM "{table_name}" ORDER BY "{col}" DESC'
                   for col in columns]

        questions = [
            f'Project the "{col}" ordered in descending order for the table {table_name}'
            for col in columns
        ]
        results = [self.database.run_query(query) for query in queries]
        sql_tags = ['ORDERBY-PROJECT'] * len(queries)
        self.append_sql_generated(sql_tags, queries, questions, results)
        return self.sql_generated
