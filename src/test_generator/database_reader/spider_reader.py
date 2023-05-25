import os

import pandas as pd

from .multiple_databases import MultipleDatabases
from .utils import (get_mask_freq_cat, CatGranularity,
                    get_sql_complexity_mask, SQLGranularity)


class SpiderReader:
    def __init__(self,
                 spider_base_path: str,
                 lower_bound: int = 1 / 3,
                 upper_bound: int = 2 / 3):
        # path where all the data is stored
        if not os.path.exists(spider_base_path):
            raise FileNotFoundError(f"Path {spider_base_path} does not exist")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.open_dbs = MultipleDatabases(spider_base_path)
        self._spider_base_path = spider_base_path
        self._tables_too_large = ['Player_Attributes', 'player', 'routes']

    @property
    def _spider_dbs_info_df(self):
        """spider df with information about the databases
        :return: df with columns
            db_id, table_names, column_names_original, column_names, table_names_original,
            column_types, foreign_keys, primary_keys"""
        df = pd.read_json(os.path.join(self._spider_base_path, "tables.json"))
        return df.set_index('db_id')

    @property
    def _spider_train_df(self):
        """consider only the questions based on single table DBs 'FROM x'
        :return: df with columns
            db_id, query, query_toks, query_toks_no_value, question, questiontoks, sql"""
        train_df = pd.read_json(os.path.join(self._spider_base_path, "train_spider.json"))
        mask_single_table = train_df.sql.map(lambda r: len(r['from']['table_units']) == 1)
        return train_df[mask_single_table]

    @property
    def _db_granularity2db_ids(self):
        """
        the col "column_types" in table.json file contains list of cols types for each DB.
        This fun. analize each DB and categorize it based on its FreqCategorical.
        :return:
         dict where
            key is the categorical frequency (Low, Medium, High)
            value is the respective list of DBs ID
        """
        df = self._spider_dbs_info_df
        mask_low_freq, mask_med_freq, mask_high_freq = get_mask_freq_cat(
            df['column_types'],
            self.lower_bound,
            self.upper_bound
        )
        return {
            CatGranularity.LOW: df.loc[mask_low_freq, :].index.tolist(),
            CatGranularity.MEDIUM: df.loc[mask_med_freq, :].index.tolist(),
            CatGranularity.HIGH: df.loc[mask_high_freq, :].index.tolist()
        }

    @property
    def _sql_granularity2mask(self):
        """
        the granularities are:
        'select': True only if SELECT x FROM y
        'simple_aggr': True only if SELECT aggr(x) FROM y where aggr is max, min, avg, ...
        'orderBy': True only if SELECT x FROM y ORDER BY z
        'where': True only if SELECT x FROM y WHERE z
        'groupBy': True only if SELECT x FROM y GROUP BY z
        'having': True only if SELECT x FROM y GROUP BY z HAVING w

        each level may contain the previous level.
        For instance the 'where' level contains all the combinations
        with or without the aggregations.
        :return: a dictionary with the masks for each sql granularity.
        """
        masks = get_sql_complexity_mask(self._spider_train_df['sql'])

        return {
            SQLGranularity.SELECT: masks['select'],
            SQLGranularity.ORDERBY: masks['orderby'],
            SQLGranularity.SIMPLE_AGGR: masks['simpleaggr'],
            SQLGranularity.WHERE: masks['where'],
            SQLGranularity.GROUPBY: masks['groupby'],
            SQLGranularity.HAVING: masks['having']
        }

    def get_df_db_granularity(self, db_level: str, df=None):
        db_level = CatGranularity[db_level]
        df = self._spider_train_df if df is None else df
        db_id_granularity: list[str] = self._db_granularity2db_ids[db_level]
        df = df[df.db_id.isin(db_id_granularity)]
        if len(df) == 0:
            raise ValueError(f"db granularity {db_level} produced empty df")
        return df

    def get_df_sql_granularity(self, sql_level: str, df=None):
        sql_level = SQLGranularity[sql_level]
        df = self._spider_train_df if df is None else df
        df = df[self._sql_granularity2mask[sql_level]]
        if len(df) == 0:
            raise ValueError(f"sql granularity {sql_level} produced empty df")
        return df

    def set_tbl_name_in_df(self, df=None):
        df = self._spider_train_df if df is None else df
        df['tbl_name'] = df.apply(
            lambda row: self.get_table_name_from_query(row.db_id, row.query),
            axis=1
        )
        df = df[df.tbl_name.notna()]
        return df

    def get_df_query_db_granularity(self, sql_granularity, cat_granularity):
        """get data where the sql complexity is sql_granularity and
         the db complexity is db_level
        :param sql_granularity: the sql complexity level

        :param cat_granularity: the db complexity level
        :return: df with columns
            db_id, query, query_toks, query_toks_no_value, question, questiontoks, sql"""
        df = self.get_df_sql_granularity(sql_granularity)
        df = self.get_df_db_granularity(cat_granularity, df=df)
        if len(df) == 0:
            raise ValueError(f"the combination db granularity {cat_granularity} "
                             f"and sql granularity {sql_granularity} "
                             f"produced empty df")

        df = self.set_tbl_name_in_df(df)
        return df

    def get_db_table_names(self, db_id) -> list[str]:
        """return all the table names associated with the db_id"""
        return self._spider_dbs_info_df.loc[db_id, 'table_names_original']

    def get_db_table(self, db_id, table, truncate=False) -> pd.DataFrame:
        """return the table associated with the db_id and table name"""
        table = self.open_dbs.get_table(db_id, table)
        return table.head(30) if truncate else table

    def get_db_query_results(self, db_id, query) -> list:
        """return query result associated with db_id"""
        return self.open_dbs.run_query(db_id, query)

    def get_table_name_from_query(self, db_id: str, query: str):
        """return the table name from the query. If too large return None"""
        table_names = self.get_db_table_names(db_id)
        tbl_name = [name for name in table_names if name.lower() in query.lower()]
        if len(tbl_name) > 0:
            tbl_name = tbl_name[0]
            return tbl_name if tbl_name not in self._tables_too_large else None
        else:
            raise ValueError(f"table name not found in query: {query}. "
                             f"Available table names in DB_id {db_id}: {table_names}")
