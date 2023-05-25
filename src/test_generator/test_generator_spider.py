from typing import Literal

import pandas as pd

from .abstract_test_generator import AbstractTestGenerator
from .database_reader.spider_reader import SpiderReader


class TestGeneratorSpider(AbstractTestGenerator):
    def __init__(self,
                 spider_base_path: str,
                 lower_bound: int = 1 / 3,
                 upper_bound: int = 2 / 3):
        self.spider_reader = SpiderReader(spider_base_path,
                                          lower_bound=lower_bound,
                                          upper_bound=upper_bound)

    def generate(self,
                 cat_granularity: Literal['LOW', 'MEDIUM', 'HIGH'],
                 sql_granularity: Literal['SELECT', 'ORDERBY', 'SIMPLE_AGGR',
                 'WHERE', 'GROUPBY', 'HAVING']
                 ) -> tuple[
        dict[tuple[str, str], pd.DataFrame],
        pd.DataFrame
    ]:
        df = self.spider_reader.get_df_query_db_granularity(
            sql_granularity=sql_granularity,
            cat_granularity=cat_granularity)

        df['query_result'] = df.apply(
            lambda row: self.spider_reader.get_db_query_results(row.db_id, row.query),
            axis=1
        )

        # 3) get the tables associated with the tests
        tbl_name2table = self._get_tables_with_names(df=df)
        return tbl_name2table, df

    def _get_tables_with_names(self, df: pd.DataFrame):
        """get the tables associated with the tests"""
        db_id_tbl_name = df.loc[:, ['db_id', 'tbl_name']].drop_duplicates().values
        tbl_name2table = {
            (db_id, tbl_name): self.spider_reader.get_db_table(db_id, tbl_name)
            for db_id, tbl_name in db_id_tbl_name
            if tbl_name is not None
        }
        return tbl_name2table
