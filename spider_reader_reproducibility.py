import os
from enum import Enum

import pandas as pd


class SQLGranularity(Enum):
    """Each level includes the previous levels"""
    SELECT = 0
    SIMPLE_AGGR = 1
    WHERE = 2
    GROUPBY = 3
    HAVING = 4
    ORDERBY = 5
    OTHER = 6


class SpiderReader:
    def __init__(self, spider_base_path: str, for_train: bool = True):
        # path where all the data is stored
        if not os.path.exists(spider_base_path):
            raise FileNotFoundError(f"Path {spider_base_path} does not exist")
        self.spider_base_path = spider_base_path
        self._tables_too_large = ['Player_Attributes', 'player', 'routes']
        self._spider_dbs_info_df = self._init_spider_dbs_info_df()
        self._spider_train_df = self._init_spider_df(for_train=for_train)
        self._sql_granularity2mask = self._init_sql_granularity2mask()

    def _init_spider_dbs_info_df(self):
        """spider df with information about the databases
        :return: df with columns
            db_id, table_names, column_names_original, column_names, table_names_original,
            column_types, foreign_keys, primary_keys"""
        df = pd.read_json(os.path.join(self.spider_base_path, "tables.json"))
        return df.set_index('db_id')

    def _init_spider_df(self, for_train):
        """consider only the questions based on single table DBs 'FROM x'
        :return: df with columns
            db_id, query, query_toks, query_toks_no_value, question, questiontoks, sql"""
        if for_train:
            df = pd.read_json(os.path.join(self.spider_base_path, "train_spider.json"))
        else:
            df = pd.read_json(os.path.join(self.spider_base_path, "dev.json"))
        mask_single_table = df.sql.map(lambda r: len(r['from']['table_units']) == 1)
        return df[mask_single_table]

    def _init_sql_granularity2mask(self):
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
        masks = self._get_sql_complexity_mask(self._spider_train_df['sql'])

        return {
            SQLGranularity.SELECT: masks['select'],
            SQLGranularity.ORDERBY: masks['orderby'],
            SQLGranularity.SIMPLE_AGGR: masks['simpleaggr'],
            SQLGranularity.WHERE: masks['where'],
            SQLGranularity.GROUPBY: masks['groupby'],
            SQLGranularity.HAVING: masks['having']
        }

    def get_df_sql_granularity(self, sql_level: str | list[str] | None, df=None):
        """return the spider tests associated with sql_level"""
        df = self._spider_train_df if df is None else df
        if sql_level is None:
            # all the sql in SQLGranularity
            sql_level = [e.name for e in SQLGranularity if e.name != 'OTHER']
        elif isinstance(sql_level, str):
            sql_level = [sql_level]

        # get the df associated with the sql_level
        dfs = []
        for level in sql_level:
            level_df = df[self._sql_granularity2mask[SQLGranularity[level]]]
            level_df.loc[:, 'sql_tags'] = level
            dfs.append(level_df)

        df = pd.concat(dfs)
        if len(df) == 0:
            raise ValueError(f"sql granularity {sql_level} produced empty df")

        df.drop(columns=['query_toks', 'query_toks_no_value', 'question_toks'],
                inplace=True)
        # add table names
        df = self.set_tbl_name_in_df(df)
        df = df.loc[:, ['db_id', 'tbl_name', 'sql_tags', 'query', 'question']]

        return df

    def get_tbl_names_in(self, db_id) -> list[str]:
        """return all the table names associated with the db_id"""
        return self._spider_dbs_info_df.loc[db_id, 'table_names_original']

    def get_table_name_from_query(self, db_id: str, query: str):
        """return the table name from the query. If too large return None"""
        table_names = self.get_tbl_names_in(db_id)
        tbl_name = [name for name in table_names if name.lower() in query.lower()]
        if len(tbl_name) > 0:
            tbl_name = tbl_name[0]
            return tbl_name if tbl_name not in self._tables_too_large else None
        else:
            raise ValueError(f"table name not found in query: {query}. "
                             f"Available table names in DB_id {db_id}: {table_names}")

    def set_tbl_name_in_df(self, df=None):
        """Set the table name from the query"""
        df = self._spider_train_df if df is None else df
        df['tbl_name'] = df.apply(
            lambda row: self.get_table_name_from_query(row.db_id, row.query),
            axis=1
        )
        df = df[df.tbl_name.notna()]
        return df

    @staticmethod
    def _get_simple_sql_mask(values: pd.Series):
        """
        Build simple mask to detect whether the SQL granularity required is present or not
        :param values: series where each element is a dict with the SQL query
            check SPIDER code to have reference
            https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql
        :return: dict with
            key: SQLGranularity
            value: simple boolean mask
        """

        def _mask_no_inner_query(sql):
            simple_where = [True]
            # avoid SQL with WHERE with inner queries
            if len(sql['where']) > 0:
                simple_where = [not isinstance(x, dict)
                                for cond in sql['where']
                                for x in cond]
            # avoid SQL with FROM with inner queries
            simple_tbl = [tbl_unit[0] != 'sql'
                          for tbl_unit in sql['from']['table_units']]
            return all(simple_where) and all(simple_tbl)

        return {
            # it is TRUE if at least one simple aggregation is present
            'SIMPLE_AGGR': values.map(lambda sql: any([x[0] != 0 for x in sql['select'][1]])),
            # it is TRUE if at least 1 WHERE is present
            'WHERE': values.map(lambda sql: len(sql['where']) > 0),
            # it is TRUE if at least 1 GROUP_BY
            'GROUPBY': values.map(lambda sql: len(sql['groupBy']) > 0),
            # it is TRUE if at least 1 HAVING
            'HAVING': values.map(lambda sql: len(sql['having']) > 0),
            # it is TRUE if at least 1 ORDER_BY
            'ORDERBY': values.map(lambda sql: len(sql['orderBy']) > 0),
            # CONSIDER ONLY NO LIMIT/INTERSECT/UNION/EXCEPT
            'NO_LIMIT_INTERSECT_UNION_EXCEPT': values.map(
                lambda s: s['limit'] == s['intersect'] == s['union'] == s['except'] is None
            ),
            # it is TRUE when NO inner query is present
            'NO_INNER_QUERY': values.map(lambda sql: _mask_no_inner_query(sql))
        }

    def _get_sql_complexity_mask(self, values: pd.Series):
        """
        Concatenate the simple SQL mask to obtain the SQL complexity masks
        :param values: series where each element is a dict with the SQL query
            check SPIDER code to have reference
            https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql
        :return: dict with
            key: SQLGranularity
            value: complex boolean mask
        """
        sql_gran2mask = self._get_simple_sql_mask(values)
        # SELECT x FROM y
        select_mask = ~sql_gran2mask['ORDERBY'] & \
                      ~sql_gran2mask['SIMPLE_AGGR'] & \
                      ~sql_gran2mask['WHERE'] & \
                      ~sql_gran2mask['GROUPBY'] & \
                      ~sql_gran2mask['HAVING'] & \
                      sql_gran2mask['NO_LIMIT_INTERSECT_UNION_EXCEPT'] & \
                      sql_gran2mask['NO_INNER_QUERY']

        # SELECT x from y order by x
        order_by_mask = sql_gran2mask['ORDERBY'] & \
                        ~sql_gran2mask['SIMPLE_AGGR'] & \
                        ~sql_gran2mask['WHERE'] & \
                        ~sql_gran2mask['GROUPBY'] & \
                        ~sql_gran2mask['HAVING'] & \
                        sql_gran2mask['NO_LIMIT_INTERSECT_UNION_EXCEPT'] & \
                        sql_gran2mask['NO_INNER_QUERY']

        # SELECT agg(x) FROM y
        simple_aggr_mask = ~sql_gran2mask['ORDERBY'] & \
                           sql_gran2mask['SIMPLE_AGGR'] & \
                           ~sql_gran2mask['WHERE'] & \
                           ~sql_gran2mask['GROUPBY'] & \
                           ~sql_gran2mask['HAVING'] & \
                           sql_gran2mask['NO_LIMIT_INTERSECT_UNION_EXCEPT'] & \
                           sql_gran2mask['NO_INNER_QUERY']

        # SELECT x FROM y WHERE z
        where_mask = ~sql_gran2mask['ORDERBY'] & \
                     ~sql_gran2mask['SIMPLE_AGGR'] & \
                     sql_gran2mask['WHERE'] & \
                     ~sql_gran2mask['GROUPBY'] & \
                     ~sql_gran2mask['HAVING'] & \
                     sql_gran2mask['NO_LIMIT_INTERSECT_UNION_EXCEPT'] & \
                     sql_gran2mask['NO_INNER_QUERY']

        # SELECT agg(z) FROM y GROUP BY z
        groupby_mask = ~sql_gran2mask['ORDERBY'] & \
                       sql_gran2mask['SIMPLE_AGGR'] & \
                       ~sql_gran2mask['WHERE'] & \
                       sql_gran2mask['GROUPBY'] & \
                       ~sql_gran2mask['HAVING'] & \
                       sql_gran2mask['NO_LIMIT_INTERSECT_UNION_EXCEPT'] & \
                       sql_gran2mask['NO_INNER_QUERY']

        # SELECT z FROM y GROUP BY z HAVING w
        having_mask = ~sql_gran2mask['ORDERBY'] & \
                      ~sql_gran2mask['SIMPLE_AGGR'] & \
                      ~sql_gran2mask['WHERE'] & \
                      sql_gran2mask['GROUPBY'] & \
                      sql_gran2mask['HAVING'] & \
                      sql_gran2mask['NO_LIMIT_INTERSECT_UNION_EXCEPT'] & \
                      sql_gran2mask['NO_INNER_QUERY']

        return {'select': select_mask,
                'orderby': order_by_mask,
                'simpleaggr': simple_aggr_mask,
                'where': where_mask,
                'groupby': groupby_mask,
                'having': having_mask}
