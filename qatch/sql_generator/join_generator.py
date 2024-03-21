from .abstract_sql_generator import AbstractSqlGenerator


class JoinGenerator(AbstractSqlGenerator):
    """
    A class for generating SQL queries, and questions based on input tables in a database.

    Attributes:
        database (SingleDatabase): The SingleDatabase object representing the database to generate queries from.
        sql_generated (dict): A dictionary containing generated SQL tags, queries, and questions.
            Format: {"sql_tags": list[str], "queries": list[str], "questions": list[str]}
    """

    def sql_generate(self, table_name: str) -> dict[str, list]:
        """
        This method generates SQL queries by performing JOIN operations on the
        specified table with other tables that have common columns in the database.
        It first empties any previously generated SQL queries, then it determines the
        tables to join based on common columns, and finally it generates the
        actual SQL queries.

        Args:
            table_name (str): The name of the input table.

        Returns:
            dict[str, list]: A dictionary where the keys are SQL tags and the values are lists of generated SQL queries.

        Examples:
             Assuming a MultipleDatabases object "database" with two tables, 'orders' and 'customers', 'orders' table has columns
            ['order_id', 'product', 'customer_id'] and 'customers' table has columns ['customer_id', 'name', 'address'].
            >>> generator = JoinGenerator(database)
            >>> generator._generate_join_project_all("orders")
            >>> generator.sql_generated
            >>> {'sql_tags': ['JOIN-PROJECT-ALL'],
            >>>    'queries': ['SELECT * FROM "orders" AS T1 JOIN customers AS T2 ON T1.customer_id=T2.customer_id'],
            >>>    'questions': ['Join all the records from table "orders" with table "customers" on "customer_id"']}

            >>> generator._generate_join_cat_columns("orders")
            >>> generator.sql_generated
            >>> {'sql_tags': ['JOIN-PROJECT-CAT'],
            >>>    'queries': ["SELECT T1.product, T2.name FROM orders AS T1 JOIN customers AS T2 ON T1.order_id=T2.order_id"],
            >>>    'questions': ['List all the "product" and "name" from the table "orders" and the table "customers" where "order_id" is the same']}

        Note:
            The method makes use of other internal methods to firstly
            get tables that can be joined with the given table and
            generate SQL queries on this basis.
        """
        self.empty_sql_generated()
        # get columns to perform the join operator that contains
        table_to_join2cols = self._get_table_name_to_join(table_name)
        self._generate_join_cat_columns(table_name, table_to_join2cols)
        self._generate_join_project_all(table_name, table_to_join2cols)
        return self.sql_generated

    def _generate_join_project_all(self, table_name: str, table_to_join2cols: dict):
        """
        A helper method to generate SQL queries, questions and SQL tags that join all records from two tables.

        This method constructs the join queries based on the given table name and a dictionary mapping tables to join columns.
        After constructing the queries, questions and SQL tags, it appends them to the sql_generated attribute using the
        append_sql_generated method.

        Args:
            table_name (str): The name of the table to be joined.
            table_to_join2cols (dict): A dictionary where the key is the name of the table to be joined
                                        and the value is a list of column names in the joining table.

        Example:
            Assuming we have two tables, 'orders' and 'customers', 'orders' table has columns
            ['order_id', 'product', 'customer_id'] and 'customers' table has columns ['customer_id', 'name', 'address'].

            >>> table_to_join2cols = {'customers': ['customer_id']}
            >>> _generate_join_project_all('orders', table_to_join2cols)

            After calling the method, the 'sql_generated' attribute of the class instance will contain the following:

            >>> {'sql_tags': ['JOIN-PROJECT-ALL'],
            >>>    'queries': ['SELECT * FROM "orders" AS T1 JOIN customers AS T2 ON T1.customer_id=T2.customer_id'],
            >>>    'questions': ['Join all the records from table "orders" with table "customers" on "customer_id"']}
        """
        queries, questions, sql_tags = [], [], []
        for t2, join_col in table_to_join2cols.items():
            for col in join_col:
                # create the join query
                queries.append(f'SELECT * FROM `{table_name}` AS T1 JOIN {t2} AS T2 ON T1.`{col}`=T2.`{col}`')
                questions.append(f'Join all the records from table "{table_name}" with table "{t2}" on "{col}"')
                sql_tags.append('JOIN-PROJECT-ALL')
        self.append_sql_generated(sql_tags, queries, questions)

    def _generate_join_cat_columns(self, table_name: str, table_to_join2cols: dict):
        """
        Helper method to generate SQL queries that joins categorical columns from two tables on a common column.
        Also generates corresponding questions and SQL tags.

        Args:
            table_name (str): The name of the base table for generating SQL queries.
            table_to_join2cols (dict): A dictionary containing table names as keys and list of common columns with base
            table as values. It indicates which tables and columns can be used for joining.

        Example:
            Assuming we have two tables, 'orders' and 'customers', 'orders' table has columns
            ['order_id', 'product', 'customer_id'] and 'customers' table has columns ['customer_id', 'name', 'address'].

            >>> table_to_join2cols = {'customers': ['customer_id']}
            >>> _generate_join_cat_columns('orders', table_to_join2cols)

            After calling the method, the 'sql_generated' attribute of the class instance will contain the following:

            >>> {'sql_tags': ['JOIN-PROJECT-CAT'],
            >>>    'queries': ["SELECT T1.product, T2.name FROM orders AS T1 JOIN customers AS T2 ON T1.order_id=T2.order_id"],
            >>>    'questions': ['List all the "product" and "name" from the table "orders" and the table "customers" where "order_id" is the same']}
        """
        queries, questions, sql_tags = [], [], []
        _, t1_cat_cols, _ = self._sample_cat_num_cols(table_name, 1)
        for t2, join_col in table_to_join2cols.items():
            _, t2_cat_cols, _ = self._sample_cat_num_cols(t2, 1)
            if not t1_cat_cols or not t2_cat_cols:
                # if there is no categorical column in the table, skip
                continue
            for col in join_col:
                # create the join query
                queries.append(f'SELECT T1.`{t1_cat_cols[0]}`, T2.`{t2_cat_cols[0]}` '
                               f'FROM `{table_name}` AS T1 JOIN {t2} AS T2 ON T1.`{col}`=T2.`{col}`')
                questions.append(
                    f'List all the "{t1_cat_cols[0]}" and "{t2_cat_cols[0]}" from the table "{table_name}" and the table "{t2}" '
                    f'where {col} is the same')
                sql_tags.append('JOIN-PROJECT-CAT')
        self.append_sql_generated(sql_tags, queries, questions)

    @staticmethod
    def _get_columns_to_join(tbl_1_cols: list, tbl_2_cols: list) -> list:
        """
        Returns the list of common columns from both tables that contain the keyword 'id' in their names.

        This method can be helpful in a SQL join operation to identify the common columns
        between two tables having 'id' keyword in their names.

        Args:
            tbl_1_cols (list): A list of column names from the first table.
            tbl_2_cols (list): A list of column names from the second table.

        Returns:
            list: A list of common column names between tbl_1_cols and tbl_2_cols,
            which contain the keyword 'id'.

        Example:
            >>> tbl_1_cols = ['user_id', 'username', 'email']
            >>> tbl_2_cols = ['product_id', 'user_id', 'product_name']
            >>> JoinGenerator._get_columns_to_join(tbl_1_cols, tbl_2_cols)
            ['user_id']
        """
        # remove all the columns that do not contain "id" in the name
        tbl_1_cols = {col for col in tbl_1_cols if "id" in col.lower()}
        tbl_2_cols = {col for col in tbl_2_cols if "id" in col.lower()}
        # get the columns that are in both tables
        cols_to_join = tbl_1_cols.intersection(tbl_2_cols)
        return list(cols_to_join)

    def _get_table_name_to_join(self, table_name: str) -> dict:
        """
        This function obtains all the tables that can be joined with the provided table based on the common columns.

        Args:
            table_name (str): The name of the table for which joining tables are to be obtained.

        Returns:
            dict: A dictionary with table names as keys and a list of common column names with the input table as values.

        Example:
            Consider three tables in the database: 'table1', 'table2', 'table3'.
            'table1' has columns 'A', 'B', 'C'. 'table2' has columns 'B', 'D', 'E'.
            'table3' has columns 'F', 'G'.
            Calling this function with 'table1' would return:
            {'table2': ['B']} as 'table2' can be joined with 'table1' on column 'B'.

        Notes:
            The function doesn't consider any inner join for now.
            It interacts with the database object associated with the class instance to obtain table names and schemas.
        """
        # get all the tables in the database
        tables = self.database.table_names
        t1_col = self.database.get_schema_given(table_name)['name']
        table_to_join = dict()
        for tbl in tables:
            if tbl == table_name:
                # skip inner join for now
                continue
            t2_col = self.database.get_schema_given(tbl)['name']
            cols_to_join = self._get_columns_to_join(t1_col, t2_col)
            if cols_to_join:
                table_to_join[tbl] = cols_to_join
        return table_to_join
