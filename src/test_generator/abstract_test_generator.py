from abc import ABC, abstractmethod

import pandas as pd


class AbstractTestGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> tuple[
        dict[tuple[str, str], pd.DataFrame],
        pd.DataFrame
    ]:
        raise NotImplementedError

    def build_test_equivalence(self, *args, **kwargs):
        """
        variate the table/question to understand whether the model is robust to change.
        :return:
        :rtype:
        """
        tbl_name2table, df = self.generate(*args, **kwargs)
        tbl_name2table_row_shuffle = {name: tbl.sample(frac=1)
                                      for name, tbl in tbl_name2table.items()}

        tbl_name2table_col_shuffle = {name: tbl.sample(frac=1, axis=1)
                                      for name, tbl in tbl_name2table.items()}

        return tbl_name2table, tbl_name2table_row_shuffle, tbl_name2table_col_shuffle, df
