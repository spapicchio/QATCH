from qatch.connectors import ConnectorTable
from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample


class JoinGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'INNER-JOIN'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Generates templates for the given ConnectorTable. The templates are created by generating
        Join Project All tests and Join Project Single tests. If the ConnectorTable has no foreign
        keys, an empty list is returned.

        Args:
            table (ConnectorTable): The ConnectorTable object for which templates will be generated.

        Returns:
            list[SingleQA]: A list of SingleQA tests that have been generated. Each SingleQA is a dictionary
                            containing a SQL query, a corresponding natural language question, and a SQL tag.

        Note:
            The returned list of tests can have varying lengths depending on the foreign keys in the input
            ConnectorTable and the category column metadata of the parent and child tables involved in the
            foreign key relationships.
        """

        if len(table.foreign_keys) == 0:
            return []
        tests = self.generate_join_project_all(table)
        tests += self.generate_join_project_single(table)
        return tests

    def generate_join_project_all(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Generates a list of SQL queries and questions representing all possible joins
        between a given table and all its child tables based on foreign key relationships.
        The join operation will use all the records from the given table and the child tables.

        Args:
            table (ConnectorTable): The table to generate join queries for.

        Returns:
            list[SingleQA]: List of dictionaries holding generated SQL queries and questions.
                            Each dictionary has the keys:
                                - 'query': The SQL query for the join operation.
                                - 'question': A description of the join operation.
                                - 'sql_tag': A tag representing the type of the SQL operation. In this case 'JOIN-PROJECT-ALL'.

        Note:
            The number of tests generated is equal to the number of foreign keys in the 'table' object.
            If there are no foreign keys, it returns an empty list.
        """

        # num of tests = len(foreign_key)
        table_name = table.tbl_name
        tests = []
        for foreign_key in table.foreign_keys:
            table_name_2 = foreign_key['child_table'].tbl_name
            parent_col = foreign_key['parent_column']
            child_col = foreign_key['child_column']
            test = SingleQA(
                query=f'SELECT * FROM `{table_name}` AS T1 '
                      f'JOIN `{table_name_2}` AS T2 ON T1.`{parent_col}` = T2.`{child_col}`',
                question=f'Join all the records from table {table_name} with table {table_name_2} on {parent_col}',
                sql_tag='JOIN-PROJECT-ALL'
            )
            tests.append(test)
        return tests

    def generate_join_project_single(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Generates a list of `SingleQA` objects based on JOIN queries derived from the input `table`.
        The JOIN queries focus on categorical columns from the parent and child tables that are connected through a foreign key.
        Each unique pair of (parent column, child column) (except for the foreign key pair) participates in a JOIN query,
        hence, the number of returned tests equals to len(foreign_key) x (len(cat_cols_parent) - 1) x (len(cat_cols_child) - 1).

        Args:
            table (ConnectorTable): The table on which to generate the JOIN queries and questions.

        Returns:
            list[SingleQA]: A list of `SingleQA` objects, each containing an SQL JOIN query, its semantic
                            representation in English, and the corresponding SQL tag.

        Note:
            - If the number of categorical columns exceeds 3 in the parent or child table, only a sample of 3
               columns are chosen for generating the JOIN queries to limit the number of generated tests.
            - If the parent or child column from the foreign key pair is the same as the currently chosen
               categorical column from parent or child table, this pair is skipped.
        """

        # num of tests = len(foreign_key) x len(cat_cols_parent) - 1  x len(cat_cols_child) - 1

        table_name = table.tbl_name
        tests = []
        cat_cols_parent = list(table.cat_col2metadata.keys())
        cat_cols_parent = utils_list_sample(cat_cols_parent, k=3, val=self.column_to_include)

        for foreign_key in table.foreign_keys:

            table_name_2 = foreign_key['child_table'].tbl_name
            parent_col = foreign_key['parent_column']
            child_col = foreign_key['child_column']
            cat_cols_child = list(foreign_key['child_table'].cat_col2metadata.keys())
            cat_cols_child = utils_list_sample(cat_cols_child, k=3, val=self.column_to_include)

            for cat_col_parent in cat_cols_parent:
                if cat_col_parent == parent_col:
                    continue

                for cat_col_child in cat_cols_child:
                    if cat_col_child == child_col:
                        continue

                    test = SingleQA(
                        query=f'SELECT T1.`{cat_col_parent}`, T2.`{cat_col_child}` '
                              f'FROM `{table_name}`'
                              f' AS T1 JOIN `{table_name_2}` AS T2 ON T1.`{parent_col}`=T2.`{child_col}`',
                        question=f'List all the {cat_col_parent} and {cat_col_child} '
                                 f'from the table {table_name} and the table {table_name_2} '
                                 f'where {parent_col} is the same',
                        sql_tag='JOIN-PROJECT-CAT'
                    )
                    tests.append(test)
        return tests
