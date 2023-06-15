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
                 cat_granularity: Literal['LOW', 'MEDIUM', 'HIGH'] | None = None,
                 sql_granularity: Literal['SELECT', 'ORDERBY', 'SIMPLE_AGGR',
                 'WHERE', 'GROUPBY', 'HAVING'] | None = None
                 ) -> tuple[
        dict[tuple[str, str], pd.DataFrame],
        pd.DataFrame
    ]:
        if cat_granularity is None or sql_granularity is None:
            df = self._generate_mult(cat_granularity, sql_granularity)
        else:
            cat_granularity = cat_granularity.upper()
            sql_granularity = sql_granularity.upper()
            df = self._generate_single(cat_granularity, sql_granularity)

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

    def _generate_single(self, cat_granularity, sql_granularity):
        df = self.spider_reader.get_df_query_db_granularity(
            sql_granularity=sql_granularity,
            cat_granularity=cat_granularity)
        df['sql_tags'] = f'{cat_granularity}-{sql_granularity}'
        return df

    def _generate_mult(self, cat_granularity, sql_granularity):
        df = pd.DataFrame()
        if cat_granularity is None and sql_granularity is None:
            # concat df for each cat_granularity and sql_granularity
            for cat_granularity in ['LOW', 'MEDIUM', 'HIGH']:
                for sql_granularity in ['SELECT', 'ORDERBY', 'SIMPLE_AGGR',
                                        'WHERE', 'GROUPBY', 'HAVING']:
                    df = pd.concat([df,
                                    self._generate_single(cat_granularity, sql_granularity)])

        elif cat_granularity is None and sql_granularity is not None:
            for cat_granularity in ['LOW', 'MEDIUM', 'HIGH']:
                df = pd.concat([df,
                                self._generate_single(cat_granularity, sql_granularity)])

        elif cat_granularity is not None and sql_granularity is None:
            for sql_granularity in ['SELECT', 'ORDERBY', 'SIMPLE_AGGR',
                                    'WHERE', 'GROUPBY', 'HAVING']:
                df = pd.concat([df,
                                self._generate_single(cat_granularity, sql_granularity)])
        return df
