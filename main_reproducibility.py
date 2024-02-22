import argparse
import json
import logging
import os

import pandas as pd
from tqdm import tqdm

from qatch import TestGenerator, MetricEvaluator
from qatch.database_reader import SingleDatabase, MultipleDatabases
from qatch.models import Tapas, Tapex, Omnitab, ChatGPT_SP, ChatGPT_QA, LLama2_QA, LLama2_SP, ChatGPT_SP_join, \
    LLama2_SP_join
from spider_reader_reproducibility import SpiderReader
from utils import read_data, save_spider_format_for_db_id

DB_IDS = ['ecommerce', 'finance', 'medicine', 'miscellaneous']

model_name2model_path = {''}


def init_model(model_name, credentials_path=None):
    model_name = model_name.lower()
    name2model = {'tapas': Tapas,
                  'tapex': Tapex,
                  'chatgpt_qa': ChatGPT_QA,
                  'chatgpt_sp': ChatGPT_SP,
                  'chatgpt_sp_join': ChatGPT_SP_join,
                  'omnitab': Omnitab,
                  'llama_qa': LLama2_QA,
                  'llama_sp': LLama2_SP,
                  'llama_sp_join': LLama2_SP_join,
                  }

    name2path = {'tapas': 'google/tapas-large-finetuned-wtq',
                 'tapex': 'microsoft/tapex-large-finetuned-wtq',
                 'omnitab': 'neulab/omnitab-large-finetuned-wtq',
                 'chatgpt_qa': 'gpt-3.5-turbo-0613',
                 'chatgpt_sp': 'gpt-3.5-turbo-0613',
                 'chatgpt_sp_join': 'gpt-3.5-turbo-0613',
                 'llama_qa': "meta-llama/Llama-2-7b-chat-hf",
                 'llama_sp': "codellama/CodeLlama-7b-Instruct-hf",
                 'llama_sp_join': "codellama/CodeLlama-7b-Instruct-hf",
                 }

    if 'chatgpt' in model_name:
        if credentials_path is None:
            raise ValueError('You need to provide the credentials path for chatGPT models')
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        model = name2model[model_name]
        model = model(api_key=credentials['api_key_chatgpt'],
                      api_org=credentials['api_org_chatgpt'],
                      model_name=name2path[model_name])
    elif 'llama' in model_name:
        if credentials_path is None:
            raise ValueError('You need to provide the credentials path for chatGPT models')
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        model = name2model[model_name]
        model = model(model_name=name2path[model_name],
                      hugging_face_token=credentials['hugging_face_token'])
    else:
        model = name2model[model_name](model_name=name2path[model_name], force_cpu=False)
    return model


def step_0_1_get_spider_data(spider_input_path):
    # spider data
    spider_reader = SpiderReader(spider_base_path=spider_input_path)
    tests_df = spider_reader.get_df_sql_granularity(sql_level=None)

    databases = MultipleDatabases(db_path=os.path.join(spider_input_path, 'database'))
    return tests_df, databases


def step_0_1_get_spider_dev_QATCH_tests(spider_input_path, seed):
    # create databases connection
    databases = MultipleDatabases(os.path.join(spider_input_path, 'database'))
    # initialize spider_dev reader
    spider_reader = SpiderReader(spider_base_path=spider_input_path, for_train=False)
    # get the database ids from the spider_dev dev set
    spider_dev_df = spider_reader.get_df_sql_granularity(sql_level=None)
    db_ids = spider_dev_df['db_id'].unique()
    # select only the database where the number of table is greater than 3
    db_ids = [db_id for db_id in db_ids if len(databases[db_id].table_names) > 3]
    # generate tests with QATCH for the selected db_ids
    test_generator = TestGenerator(databases=databases)
    tests_df = test_generator.generate(generators=None, db_names=db_ids, seed=seed)
    # remove tables too large
    table_len_mask = tests_df.apply(lambda row:
                                    True
                                    if databases[row.db_id].get_table_given(row.tbl_name).size < 500
                                    else False,
                                    axis=1)
    tests_df = tests_df.loc[table_len_mask, :]
    return tests_df.reset_index(drop=True), databases


def step_0_get_proprietary_data(model_name, base_path_input, inject_null_percentage, db_save_path, seed):
    db_save_path = os.path.join(db_save_path, model_name)
    for db_id in DB_IDS:
        db_tables: dict[str, pd.DataFrame] = read_data(db_id=db_id,
                                                       model_name=model_name,
                                                       input_base_path_data=base_path_input,
                                                       seed=seed,
                                                       inject_null_percentage=inject_null_percentage)

        # create database connection
        _ = SingleDatabase(db_path=db_save_path, db_name=db_id, tables=db_tables)
    databases = MultipleDatabases(db_save_path)
    return databases


def step_1_generate_tests(databases, seed):
    # init generator
    test_generator = TestGenerator(databases=databases)
    # generate tests for each database and for each generator
    tests_df = test_generator.generate(generators=None, db_names=None, seed=seed)
    return tests_df


def step_2_run_tests(tests_df, databases, model):
    tqdm.pandas(desc=f'Predicting for {model.name}')
    tests_df[f'predictions_{model.name}'] = tests_df.progress_apply(lambda row: model.predict(
        table=databases.get_table(db_id=row['db_id'], tbl_name=row['tbl_name']) if isinstance(row['tbl_name'], str) else None,
        query=row['question'],
        tbl_name=row['tbl_name'],
        db_table_schema=databases.get_all_table_schema_given(db_id=row['db_id'])
    ), axis=1)

    return tests_df


def step_3_evaluate_tests(tests_df, databases, task, prediction_col_name):
    evaluator = MetricEvaluator(databases=databases)
    tests_df = evaluator.evaluate_with_df(tests_df,
                                          prediction_col_name=prediction_col_name,
                                          task=task,
                                          # keep_target=True
                                          keep_target=False
                                          )
    return tests_df


def save_df(args, tests_df, databases, for_train):
    if args.generate_tests_for == 'spider_dev':
        if for_train:
            test_file_name = f'{args.task.upper()}_{args.model_name}_spider_train_df.json'
        else:
            test_file_name = f'{args.task.upper()}_{args.model_name}_spider_dev_df.json'

        tests_df.to_json(os.path.join(args.db_save_path, test_file_name),
                         orient='records')
    elif args.generate_tests_for == 'spider_dev_qatch':
        test_file_name = f'{args.task.upper()}_{args.model_name}_spider_dev_qatch_test_df.json'
        tests_df.to_json(os.path.join(args.db_save_path, test_file_name),
                         orient='records')
    else:
        tables_all_db = []
        for db_id in databases.get_db_names():
            _, table_db = save_spider_format_for_db_id(df=tests_df[tests_df.db_id == db_id],
                                                       db=databases[db_id],
                                                       model_name=args.model_name)
            tables_all_db.append(table_db)

        tables_all_db = pd.concat(tables_all_db).reset_index(drop=True)
        tests_df.to_json(os.path.join(args.db_save_path, args.model_name,
                                      f'{args.task.upper()}_{args.model_name}_tests_df.json'),
                         orient='records')
        tables_all_db.to_json(os.path.join(args.db_save_path, args.model_name,
                                           f'{args.task.upper()}_{args.model_name}_all_dbs_tables.json'),
                              orient='records')


def reproduce_experiments(args):
    model = init_model(args.model_name, args.credentials_path)
    if args.generate_tests_for == 'spider_dev':
        # spider_dev data
        tests_df, databases = step_0_1_get_spider_data(spider_input_path=args.spider_input_path,
                                                       for_train=args.is_spider_dev)
    elif args.generate_tests_for == 'spider_dev_qatch':
        # spider_dev dev data but with QATCH tests
        tests_df, databases = step_0_1_get_spider_dev_QATCH_tests(spider_input_path=args.spider_input_path,
                                                                  seed=args.seed)
    else:
        # proprietary data
        # step 0
        databases = step_0_get_proprietary_data(model_name=args.model_name,
                                                base_path_input=args.propr_input_path,
                                                inject_null_percentage=args.inject_null_percentage,
                                                db_save_path=args.db_save_path,
                                                seed=args.seed)
        # step 1
        tests_df = step_1_generate_tests(databases=databases, seed=args.seed)
        databases = MultipleDatabases('results_29_11_2023/proprietary/QA/tapex')
    # save_df(args=args, tests_df=tests_df, databases=databases, for_train=args.is_spider_dev)
    # step 2
    tests_df = step_2_run_tests(tests_df=tests_df, databases=databases, model=model)
    save_df(args=args, tests_df=tests_df, databases=databases, for_train=args.is_spider_dev)
    # step 3
    tests_df = step_3_evaluate_tests(tests_df=tests_df,
                                     databases=databases,
                                     task=args.task,
                                     prediction_col_name=f'predictions_{model.name}')
    save_df(args=args, tests_df=tests_df, databases=databases, for_train=args.is_spider_dev)


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(prog='main_reproducibility.py',
                                     description='Used to reproduce the experiments for QATCH')
    parser.add_argument('--task', default='QA', help='The task to evaluate. '
                                                     'supported tasks: [QA, SP]')

    parser.add_argument('-gtf', '--generate_tests_for', default='proprietary',
                        help='the type of tests to generate.\n' 'supported types: '
                             '[proprietary, spider_dev, spider_dev_qatch]')

    parser.add_argument('--is_spider_dev', action='store_false',
                        help='if True generate tests for the spider_dev dev set')

    parser.add_argument('--model_name',
                        help='Model name used to define the length of the input table,\n'
                             'supported QA models = [tapas, tapex, omnitab, chatGPT-QA, LLama-QA]\n'
                             'supported SP models = [data, gap, skg, chatGPT-SP, LLama-SP]')

    parser.add_argument('--propr_input_path', type=str, default='./data',
                        help='the base path where the proprietary data is stored. '
                             'Default is data')

    parser.add_argument('--spider_input_path', type=str, default='./data/spider_dev',
                        help='the base path where the spider_dev data is stored. '
                             'Default is data')

    parser.add_argument('-dsp', '--db_save_path', type=str, required=True,
                        help='the path where the generated tests will be stored.')

    parser.add_argument("--inject_null_percentage", type=float, required=True,
                        help='percentage of null to inject in the tables before running tests')

    parser.add_argument("--seed", type=int, default=2023,
                        help='seed used to generate the tests for reproducibility')

    parser.add_argument("--credentials_path", type=str, default=None,
                        help='path to the credentials file for chatGPT models')

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    args = parser.parse_args()
    args.seed = int(args.seed)
    if args.model_name in ["data", "gap", "skg", "chatGPT-SP", "LLama-SP"]:
        if args.task != "SP":
            raise ValueError("The model name is not compatible with the task")
    if args.verbose:
        params_dict = vars(args)
        logging.basicConfig(level=logging.INFO)
        logging.info('Script arguments:')
        [logging.info(f'\t{key}: {value}') for key, value in params_dict.items()]

    reproduce_experiments(args)
