import argparse
import json
import logging
import pickle

import pandas as pd
from tqdm import tqdm

from src import Tapas, Tapex, ChatGPT, Omnitab


def init_model(model_name, model_path, credentials_path=None):
    model_name = model_name.lower()
    name2model = {'tapas': Tapas, 'tapex': Tapex, 'chatgpt-qa': ChatGPT,
                  'chatgpt-sp': ChatGPT, 'omnitab': Omnitab}

    if 'chatgpt' in model_name:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        model = ChatGPT(api_key=credentials['api_key_chatgpt'],
                        api_org=credentials['api_org_chatgpt'],
                        test_type=model_name.split('-')[-1])
    else:
        model = name2model[model_name](model_path=model_path)

    return model


def get_table(tables, row):
    # check whether the key of the dictionary is a tuple or a string
    if isinstance(list(tables.keys())[0], str):
        table = tables[row.tbl_name]
    else:
        table = tables[(row.db_id, row.tbl_name)]
    return table


def main(args):
    model = init_model(args.model_name, args.model_path, args.credentials_path_chatgpt)

    input_df = pd.read_json(args.tests_input_path)
    with open(args.tests_input_tables_path, 'rb') as f:
        tables = pickle.load(f)

    tqdm.pandas(desc=f'Predicting tests for {args.model_name}')
    col_pred = f'prediction_{args.model_name}'
    input_df[col_pred] = input_df.progress_apply(
        lambda row: model.predict(table=get_table(tables, row),
                                  queries=row.question,
                                  tbl_name=row.tbl_name),
        axis=1
    )
    input_df.to_json(args.output_df_path_json, orient='records')
    if args.verbose:
        logging.info(f'Predictions saved in {args.output_df_path_json}')
        logging.info(f'Example predictions:')
        logging.info(input_df[col_pred][:3])
    return input_df


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(prog='02_predict_tests.py',
                                     description='The scripts is used to predict the tests for the proprietary '
                                                 'and spider datasets')

    parser.add_argument('--model_name',
                        help='model name used to define the length of the input table,\n'
                             'supported QA models = [tapas, tapex, omnitab, chatGPT-QA]\n'
                             'supported SP models = [chatGPT-SP]')

    parser.add_argument('--model_path',
                        help='for Tapas and Tapex, you need to specify the huggingface path',
                        default=None)

    parser.add_argument('--tests_input_path', help='the path to the input tests file',
                        required=True)

    parser.add_argument('--tests_input_tables_path',
                        help='the path to the input tables expected pickle file',
                        required=True)
    parser.add_argument('--output_df_path_json', help='The json file path to store the df with the predictions',
                        required=True)
    parser.add_argument('--input_col_name', help='the column name used as input for the model',
                        default='question')

    parser.add_argument('--credentials_path_chatgpt', help='the path to the credentials json file for chatpgt',
                        default='credentials.json')

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    args = parser.parse_args()
    if args.model_name in ['tapas', 'tapex', 'omnitab'] and args.model_path is None:
        raise ValueError(f'You need to specify the model path taken from huggingface '
                         f'for {args.model_name}')

    if 'chatgpt' in args.model_name and args.credentials_path_chatgpt is None:
        raise ValueError(f'You need to specify the credentials path for chatgpt')

    if args.verbose:
        params_dict = vars(args)
        logging.basicConfig(level=logging.INFO)
        logging.info('Script arguments:')
        [logging.info(f'\t{key}: {value}') for key, value in params_dict.items()]

    main(args)
