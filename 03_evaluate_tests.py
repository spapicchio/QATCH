import argparse
import re

import pandas as pd

from runners.utils import get_predictions_results_from_dbs
from src import MetricEvaluator


def substitute_placeholders(query, predictions):
    pattern = r'[\'](.*?)[\']'  # Matches any string within single quotes
    string_values = re.findall(pattern, query)

    for idx, value in enumerate(string_values):
        predictions = predictions.replace("terminal", f"{value}", 1)
    return predictions


def main(args):
    predictions_col = f'predictions-{args.model_name}'
    # 1. Read tests
    tests_df = pd.read_json(args.tests_path)
    tests_df['query_result'] = tests_df.query_result.map(lambda x: eval(x))

    if args.task == 'SP':
        # 2. run the query on the DBs only for SP
        query_result_col_name = f'query_result_predictions_{args.model_name}'
        tests_df = get_predictions_results_from_dbs(base_path_db=args.tests_db_path,
                                                    df=tests_df,
                                                    prediction_col_name=predictions_col,
                                                    query_result_col_name=query_result_col_name)
    else:
        query_result_col_name = predictions_col

    # 3. init the metric evaluator
    metric_evaluator = MetricEvaluator(
        metrics=['cell_precision', 'cell_recall', 'tuple_cardinality', 'tuple_constraint', 'tuple_order']
    )
    # 4. evaluate the results between the query results and the predictions
    tests_df = metric_evaluator.evaluate_with_df(tests_df,
                                                 target='query_result',
                                                 predictions=query_result_col_name,
                                                 task=args.task)
    tests_df.rename(columns={
        'cell_precision': f'cell_precision_{args.model_name}',
        'cell_recall': f'cell_recall_{args.model_name}',
        'tuple_cardinality': f'tuple_cardinality_{args.model_name}',
        'tuple_constraint': f'tuple_constraint_{args.model_name}',
        'tuple_order': f'tuple_order_{args.model_name}',
    }, inplace=True)

    if args.task == 'SP':
        tests_df.drop(columns=query_result_col_name, inplace=True)
    return tests_df


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(prog='03_evaluate_tests.py',
                                     description='the scripts evaluates tests results for QA and SP')
    parser.add_argument('--model_name',
                        required=True,
                        help='model name used to define the length of the input table,\n'
                             'supported QA models = [tapas, tapex, omnitab, chatGPT-QA]\n'
                             'supported SP models = [resdsql, gap, skg, chatGPT-SP]')

    parser.add_argument('--tests_path',
                        help='the path to the tests json file')
    parser.add_argument('--tests_db_path', '-tdp',
                        required=True,
                        help='the path to the database generated from the tests')
    parser.add_argument('--task',
                        required=True,
                        help='the task to evaluate, supported tasks = [QA, SP]')
    parser.add_argument('--save_path', help='the path to save the results')

    args = parser.parse_args()
    df = main(args)
    args.tests_path = args.save_path
    df.reset_index(drop=True).to_json(args.save_path, orient='records')
