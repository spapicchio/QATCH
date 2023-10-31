import argparse
import logging
import os
import pickle

import pandas as pd

from runners.utils import read_data
from src import TestGenerator
from src import TestGeneratorSpider


def get_tests_propr(args, db_id):
    # get pandas dataframe
    db_tables = read_data(db_id=db_id,
                          model_name=args.model_name,
                          input_base_path_data=args.propr_input_path,
                          seed=args.seed,
                          inject_null_percentage=args.inject_null_percentage)
    # init tests generator and save to db the pandas dataframes
    test_generator = TestGenerator(db_save_path=args.db_save_path,
                                   db_tables=db_tables,
                                   db_name=db_id,
                                   seed=args.seed)
    # generate tests
    tables, tests_df = test_generator.generate(generators=args.sql_generators,
                                               table_names=args.tbl_names,
                                               save_spider_format=True)
    with open(os.path.join(args.db_save_path, db_id, f'db_table_{db_id}.pickle'), 'wb') as handle:
        pickle.dump(tables, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tables, tests_df


def generate_proprietary_tests(args):
    db_id_list = args.db_id
    proprietary_path_tests = os.path.join(args.db_save_path, 'proprietary_tests.json')
    proprietary_path_tables = os.path.join(args.db_save_path, 'db_tables_proprietary.pickle')
    if os.path.exists(proprietary_path_tests):
        tests_df = pd.read_json(proprietary_path_tests)
        tables = pickle.load(open(proprietary_path_tables, 'rb'))
    else:
        if not isinstance(db_id_list, list):
            db_id_list = [db_id_list]

        tables_tests = [get_tests_propr(args, db_id) for db_id in db_id_list]

        tests_df = pd.concat([tests_df for (_, tests_df) in tables_tests])
        tables = [tables for (tables, _) in tables_tests]

        tests_df.reset_index(drop=True).to_json(proprietary_path_tests, orient='records')
        table_dict = {}
        for table in tables:
            table_dict.update(table)
        with open(proprietary_path_tables, 'wb') as handle:
            pickle.dump(table_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tables, tests_df


def generate_spider_tests(spider_base_path, db_save_path):
    spider_path_tables = os.path.join(db_save_path, 'db_tables_spider.pickle')
    spider_path_tests = os.path.join(db_save_path, 'spider_tests.json')
    if os.path.exists(spider_path_tests):
        tests_df = pd.read_json(spider_path_tests)
        tables = pickle.load(open(spider_path_tables, 'rb'))
    else:
        test_generator = TestGeneratorSpider(spider_base_path)
        tables, tests_df = test_generator.generate()
        if not os.path.exists(db_save_path):
            os.makedirs(db_save_path)
        tests_df.to_json(spider_path_tests, orient='records')
        with open(spider_path_tables, 'wb') as handle:
            pickle.dump(tables, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tables, tests_df


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(prog='01_generate_tests.py',
                                     description='The scripts generates tests for '
                                                 'proprietary and spider datasets')

    parser.add_argument('-gtf', '--generate_tests_for', default='proprietary',
                        help='the type of tests to generate.\n' 'supported types: [proprietary, spider]')

    parser.add_argument('--model_name',
                        help='model name used to define the length of the input table,\n'
                             'supported QA models = [tapas, tapex, omnitab, chatGPT-QA]\n'
                             'supported SP models = [resdsql, gap, skg, chatGPT-SP]')

    parser.add_argument('--db_id', default=None,
                        nargs='*',
                        help='the proprietary db_id used to generate the tests.\n'
                             'Accepted categories [ecommerce, finance, medicine, miscellaneous]')

    parser.add_argument('--tbl_names', default=None, nargs='*',
                        help='the proprietary tables in the selected category. '
                             'Only used if you want to test a specific table.')

    parser.add_argument('--sql_generators', default=None, nargs='*',
                        help='the sql generators to use for test generation.\n'
                             'supported generators: [select, orderby, distinct, where, groupby,'
                             'having, simpleAgg, nullCount]')

    parser.add_argument('--propr_input_path', type=str, default='data',
                        help='the base path where the proprietary data is stored. '
                             'Default is data')

    parser.add_argument('--spider_input_path', type=str, default='data/spider',
                        help='the base path where the spider data is stored. '
                             'Default is data')

    parser.add_argument('-dsp', '--db_save_path', type=str, required=True,
                        help='the path where the generated tests will be stored.')

    parser.add_argument("--inject_null_percentage", type=float, required=True,
                        help='percentage of null to inject in the tables before running tests')

    parser.add_argument("--seed", type=float, default=2023,
                        help='seed used to generate the tests for reproducibility')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    args = parser.parse_args()
    if args.db_id is None:
        args.db_id = ['ecommerce', 'finance', 'medicine', 'miscellaneous']
    if args.verbose:
        params_dict = vars(args)
        logging.basicConfig(level=logging.INFO)
        logging.info('Script arguments:')
        [logging.info(f'\t{key}: {value}') for key, value in params_dict.items()]

    if args.generate_tests_for == 'spider':
        generate_spider_tests(args.spider_input_path, args.db_save_path)
    elif args.generate_tests_for == 'proprietary':
        generate_proprietary_tests(args)
    else:
        raise ValueError(f'{args.generate_tests_for} is not a supported test type. '
                         f'Please use one of the following: [proprietary, spider]')
