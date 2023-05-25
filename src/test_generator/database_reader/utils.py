from collections import Counter
from enum import Enum

import pandas as pd


class CatGranularity(Enum):
    """defines the presence of categorical columns in the table"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class SQLGranularity(Enum):
    """Each level includes the previous levels"""
    SELECT = 0
    SIMPLE_AGGR = 1
    WHERE = 2
    GROUPBY = 3
    HAVING = 4
    ORDERBY = 5
    OTHER = 6


def get_mask_freq_cat(values: pd.Series, lower_bound: float, upper_bound: float):
    """
    return three masks based on the number of times the values 'text' is present in the
    values of the series.
    :param values: series where each element is a lis: [text, text, float, float, ...]
    :param lower_bound: lower bound for mask_low
    :param upper_bound: upper bound for mask_high
    :return: three boolean masks based on the number of times the values 'text' is present
    """
    mask_low = values.map(lambda r: Counter(r)['text'] <= len(r) * lower_bound)
    mask_medium = values.map(
        lambda r: len(r) * lower_bound < Counter(r)['text'] < len(r) * upper_bound
    )
    mask_high = values.map(lambda r: Counter(r)['text'] >= len(r) * upper_bound)
    return mask_low, mask_medium, mask_high


def get_simple_sql_mask(values: pd.Series):
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
            simple_where = [not isinstance(x, dict) for cond in sql['where'] for x in
                            cond]
        # avoid SQL with FROM with inner queries
        simple_tbl = [tbl_unit[0] != 'sql' for tbl_unit in sql['from']['table_units']]
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


def get_sql_complexity_mask(values: pd.Series):
    """
    Concatenate the simple SQL mask to obtain the SQL complexity masks
    :param values: series where each element is a dict with the SQL query
        check SPIDER code to have reference
        https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql
    :return: dict with
        key: SQLGranularity
        value: complex boolean mask
    """
    sql_gran2mask = get_simple_sql_mask(values)
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
