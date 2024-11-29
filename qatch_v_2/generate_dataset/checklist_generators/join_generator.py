from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample
from qatch_v_2.connectors import ConnectorTable


class JoinGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'INNER-JOIN'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        if len(table.foreign_keys) == 0:
            return []
        tests = self.generate_join_project_all(table)
        tests += self.generate_join_project_single(table)
        return tests

    def generate_join_project_all(self, table: ConnectorTable) -> list[SingleQA]:
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
        # num of tests = len(foreign_key) x len(cat_cols_parent) - 1  x len(cat_cols_child) - 1

        table_name = table.tbl_name
        tests = []
        cat_cols_parent = table.cat_col2metadata.keys()
        cat_cols_parent = utils_list_sample(cat_cols_parent, k=3)

        for foreign_key in table.foreign_keys:

            table_name_2 = foreign_key['child_table'].tbl_name
            parent_col = foreign_key['parent_column']
            child_col = foreign_key['child_column']
            cat_cols_child = foreign_key['child_table'].cat_col2metadata.keys()
            cat_cols_child = utils_list_sample(cat_cols_child, k=3)

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
