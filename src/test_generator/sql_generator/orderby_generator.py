from .abstract_sql_generator import AbstractSqlGenerator
from ..database_reader import SingleDatabase


class OrderByGenerator(AbstractSqlGenerator):

    def __int__(self, database: SingleDatabase, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

    def sql_generate(self, table_name: str):
        # TODO include order of one column with one select
        # TODO include order with multiple columns (?)
        columns = self.database.get_columns_from_table(table_name)

        queries, questions, results = [], [], []
        # create a question, query with all the combinations of columns
        for order in ['ASC', 'DESC']:
            queries += [
                f'SELECT * FROM "{table_name}" ORDER BY "{col}" {order}'
                for col in columns
            ]

            questions += [
                f'Show all data ordered by {col} in' \
                f' {"ascending" if order.lower() == "asc" else "descending"} ' \
                f'order for the table {table_name}'

                for col in columns
            ]

        # run the query and get the results
        results = [self.database.run_query(query) for query in queries]
        sql_tags = ['ORDERBY-SINGLE'] * len(queries)
        return {'sql_tags': sql_tags, 'queries': queries,
                'questions': questions, 'results': results}
