from .abstract_sql_generator import AbstractSqlGenerator


class JoinGenerator(AbstractSqlGenerator):
    def sql_generate(self, table_name: str) -> dict[str, list]:
        self.empty_sql_generated()
        # get columns to perform the join operator that contains
        table_to_join2cols = self._get_table_name_to_join(table_name)
        self._generate_join_cat_columns(table_name, table_to_join2cols)
        self._generate_join_project_all(table_name, table_to_join2cols)
        return self.sql_generated

    def _generate_join_project_all(self, table_name: str, table_to_join2cols: dict):
        queries, questions, sql_tags = [], [], []
        for t2, join_col in table_to_join2cols.items():
            for col in join_col:
                # create the join query
                queries.append(f'SELECT * FROM "{table_name}" AS T1 JOIN {t2} AS T2 ON T1.{col}=T2.{col}')
                questions.append(f'Join all the records from table "{table_name}" with table "{t2}" on "{col}"')
                sql_tags.append('JOIN-PROJECT-ALL')
        self.append_sql_generated(sql_tags, queries, questions)

    def _generate_join_cat_columns(self, table_name: str, table_to_join2cols: dict):
        queries, questions, sql_tags = [], [], []
        _, t1_cat_cols, _ = self._sample_cat_num_cols(table_name, 1)
        for t2, join_col in table_to_join2cols.items():
            _, t2_cat_cols, _ = self._sample_cat_num_cols(t2, 1)
            if not t1_cat_cols or not t2_cat_cols:
                # if there is no categorical column in the table, skip
                continue
            for col in join_col:
                # create the join query
                queries.append(f'SELECT T1.{t1_cat_cols[0]}, T2."{t2_cat_cols[0]}" '
                               f'FROM "{table_name}" AS T1 JOIN {t2} AS T2 ON T1.{col}=T2.{col}')
                questions.append(
                    f'List all the "{t1_cat_cols[0]}" and "{t2_cat_cols[0]}" from the table "{table_name}" and the table "{t2}" '
                    f'where {col} is the same')
                sql_tags.append('JOIN-PROJECT-CAT')
        self.append_sql_generated(sql_tags, queries, questions)

    @staticmethod
    def _get_columns_to_join(tbl_1_cols: list, tbl_2_cols: list) -> list:
        """return the columns that are in both tables and contains id in the name"""
        # remove all the columns that do not contain "id" in the name
        tbl_1_cols = {col for col in tbl_1_cols if "id" in col.lower()}
        tbl_2_cols = {col for col in tbl_2_cols if "id" in col.lower()}
        # get the columns that are in both tables
        cols_to_join = tbl_1_cols.intersection(tbl_2_cols)
        return list(cols_to_join)

    def _get_table_name_to_join(self, table_name: str) -> dict:
        """return all the tables that can be joined with the given table"""
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
