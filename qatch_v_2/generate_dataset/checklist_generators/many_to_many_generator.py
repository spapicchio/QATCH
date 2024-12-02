from .base_generator import BaseGenerator, SingleQA
from ...connectors import ConnectorTable


class ManyToManyGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'many-to-many-generator'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        # given n = len(cat_cols)
        # len of tests: n*(n-1)/2.
        cat_cols = list(table.cat_col2metadata.keys())
        tests = []

        for i in range(len(cat_cols) - 1):
            for j in range(i + 1, len(cat_cols)):
                selected_cols = [cat_cols[i], cat_cols[j]]
                single_test = SingleQA(
                    query=f"SELECT `{selected_cols[0]}` FROM `{table.tbl_name}`"
                          f" GROUP BY `{selected_cols[0]}` HAVING COUNT(DISTINCT `{selected_cols[1]}`) = ("
                          f"SELECT COUNT(DISTINCT `{selected_cols[1]}`) FROM `{table.tbl_name}`"
                          f")",
                    question=f"What are the {selected_cols[0]} with all the {selected_cols[1]} "
                             f"in table {table.tbl_name}?",
                    sql_tag='many-to-many'
                )
                tests.append(single_test)
        return tests
