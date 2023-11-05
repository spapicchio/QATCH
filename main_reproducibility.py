import argparse
import json
import logging
import os

import pandas as pd
from tqdm import tqdm

from qatch import TestGenerator, MetricEvaluator
from qatch.database_reader import SingleDatabase, MultipleDatabases
from qatch.models import Tapas, Tapex, Omnitab, ChatGPT_SP, ChatGPT_QA, LLama2_QA, LLama2_SP
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
                  'omnitab': Omnitab,
                  'llama_qa': LLama2_QA,
                  'llama_sp': LLama2_SP}

    name2path = {'tapas': 'google/tapas-large-finetuned-wtq',
                 'tapex': 'microsoft/tapex-large-finetuned-wtq',
                 'omnitab': 'neulab/omnitab-large-finetuned-wtq',
                 'chatgpt_qa': 'gpt-3.5-turbo-0613',
                 'chatgpt_sp': 'gpt-3.5-turbo-0613',
                 'llama_qa': 'meta-llama/Llama-2-7b-chat-hf',
                 'llama_sp': 'meta-llama/Llama-2-7b-chat-hf'}

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
        model = name2model[model_name](model_name=name2path[model_name])

    return model


def step_0_1_get_spider_data(spider_input_path):
    # spider data
    spider_reader = SpiderReader(spider_base_path=spider_input_path)
    tests_df = spider_reader.get_df_sql_granularity(sql_level=None)
    databases = MultipleDatabases(db_path=os.path.join(spider_input_path, 'database'))
    return tests_df, databases


def step_0_get_proprietary_data(model_name, base_path_input, inject_null_percentage, db_save_path):
    db_save_path = os.path.join(db_save_path, model_name)
    for db_id in DB_IDS:
        db_tables: dict[str, pd.DataFrame] = read_data(db_id=db_id,
                                                       model_name=model_name,
                                                       input_base_path_data=base_path_input,
                                                       seed=2023,
                                                       inject_null_percentage=inject_null_percentage)

        # create database connection
        db = SingleDatabase(db_path=db_save_path, db_name=db_id, tables=db_tables)
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
        table=databases.get_table(db_id=row['db_id'], tbl_name=row['tbl_name']),
        query=row['question'],
        tbl_name=row['tbl_name']
    ), axis=1)

    return tests_df


def step_3_evaluate_tests(tests_df, databases, task, prediction_col_name):
    evaluator = MetricEvaluator(databases=databases)
    tests_df = evaluator.evaluate_with_df(tests_df,
                                          prediction_col_name=prediction_col_name,
                                          task=task)
    return tests_df


def reproduce_experiments(args):
    model = init_model(args.model_name, args.credentials_path)
    if args.generate_tests_for == 'spider':
        # spider data
        tests_df, databases = step_0_1_get_spider_data(spider_input_path=args.spider_input_path)
    else:
        # proprietary data
        # step 0
        databases = step_0_get_proprietary_data(model_name=args.model_name,
                                                base_path_input=args.propr_input_path,
                                                inject_null_percentage=args.inject_null_percentage,
                                                db_save_path=args.db_save_path)
        # step 1
        tests_df = step_1_generate_tests(databases=databases, seed=args.seed)


    # step 2
    tests_df = step_2_run_tests(tests_df=tests_df, databases=databases, model=model)
    save_df(args=args, tests_df=tests_df, databases=databases)

    # step 3
    tests_df = step_3_evaluate_tests(tests_df=tests_df,
                                     databases=databases,
                                     task=args.task,
                                     prediction_col_name=f'predictions_{model.name}')
    save_df(args=args, tests_df=tests_df, databases=databases)


def save_df(args, tests_df, databases):
    if args.generate_tests_for == 'spider':
        test_file_name = f'{args.task.upper()}_{args.model_name}_spider_train_test_results.json'
        tests_df.to_json(os.path.join(args.db_save_path, test_file_name),
                         orient='records')

    else:
        test_all_db, tables_all_db = [], []
        for db_id in databases.get_db_names():
            df_db, table_db = save_spider_format_for_db_id(df=tests_df[tests_df.db_id == db_id],
                                                           db=databases[db_id],
                                                           model_name=args.model_name)
            test_all_db.append(df_db)
            tables_all_db.append(table_db)
        test_all_db = pd.concat(test_all_db).reset_index(drop=True)
        tables_all_db = pd.concat(tables_all_db).reset_index(drop=True)
        test_all_db.to_json(os.path.join(args.db_save_path,
                                         f'{args.task.upper()}_{args.model_name}_all_dbs_results.json'),
                            orient='records')
        tables_all_db.to_json(os.path.join(args.db_save_path,
                                           f'{args.task.upper()}_{args.model_name}_all_dbs_tables.json'),
                              orient='records')


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(prog='main_reproducibility.py',
                                     description='Used to reproduce the experiments for QATCH')
    parser.add_argument('--task', default='QA', help='The task to evaluate. '
                                                     'supported tasks: [QA, SP]')

    parser.add_argument('-gtf', '--generate_tests_for', default='proprietary',
                        help='the type of tests to generate.\n' 'supported types: [proprietary, spider]')

    parser.add_argument('--model_name',
                        help='Model name used to define the length of the input table,\n'
                             'supported QA models = [tapas, tapex, omnitab, chatGPT-QA, LLama-QA]\n'
                             'supported SP models = [resdsql, gap, skg, chatGPT-SP, LLama-SP]')

    parser.add_argument('--propr_input_path', type=str, default='./data',
                        help='the base path where the proprietary data is stored. '
                             'Default is data')

    parser.add_argument('--spider_input_path', type=str, default='./data/spider',
                        help='the base path where the spider data is stored. '
                             'Default is data')

    parser.add_argument('-dsp', '--db_save_path', type=str, required=True,
                        help='the path where the generated tests will be stored.')

    parser.add_argument("--inject_null_percentage", type=float, required=True,
                        help='percentage of null to inject in the tables before running tests')

    parser.add_argument("--seed", type=float, default=2023,
                        help='seed used to generate the tests for reproducibility')

    parser.add_argument("--credentials_path", type=str, default=None,
                        help='path to the credentials file for chatGPT models')

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    args = parser.parse_args()
    if args.model_name in ["resdsql", "gap", "skg", "chatGPT-SP", "LLama-SP"]:
        if args.task != "SP":
            raise ValueError("The model name is not compatible with the task")
    if args.verbose:
        params_dict = vars(args)
        logging.basicConfig(level=logging.INFO)
        logging.info('Script arguments:')
        [logging.info(f'\t{key}: {value}') for key, value in params_dict.items()]

    reproduce_experiments(args)
