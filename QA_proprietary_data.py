import argparse
import logging

from tqdm import tqdm

from utils import init_model, init_test_generator, get_summary, init_metric_evaluator


def main(args):
    if args.verbose:
        params_dict = vars(args)
        logging.basicConfig(level=logging.INFO)
        logging.info('Script arguments:')
        [logging.info(f'\t{key}: {value}') for key, value in params_dict.items()]

    # init QATCH components
    model = init_model(args.model_name)
    test_generator = init_test_generator(args.db_category,
                                         args.model_name,
                                         args.output_db_path,
                                         args.input_base_path)
    metric_evaluator = init_metric_evaluator(args.metrics)

    # 1. QATCH-Generate
    if args.verbose:
        logging.info('Starting generating tests')
    tables, tests_df = test_generator.generate(generators=args.sql_generators,
                                               table_names=args.tbl_names)
    # 2. TRL model predictions
    if args.verbose:
        logging.info('Starting predictions')
    tqdm.pandas(desc='predictions')
    tests_df['predictions'] = tests_df.progress_apply(
        lambda row: model.predict(table=tables[row.tbl_name], queries=row.question),
        axis=1
    )

    # 3. QATCH-Evaluate
    if args.verbose:
        logging.info('Starting evaluating tests')
    tests_df = metric_evaluator.evaluate_with_df(tests_df,
                                                 target='query_result',
                                                 predictions='predictions')
    # 4. save tests results and summary

    path = f'{args.output_db_path}/{args.db_category}/tests_with_results.json'
    tests_df.reset_index(drop=True).to_json(path)
    if args.verbose:
        logging.info(f'Saved tests in {path}')
    results = get_summary(tests_df, args.metrics)
    path = f'{args.output_db_path}/{args.db_category}/summary.csv'
    results.to_csv(path)
    if args.verbose:
        logging.info(f'Saved summary in {path}')


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(prog='QA_proprietary_data',
                                     description='Run question answering for proprietary data')
    parser.add_argument('--model_name', type=str, default='google/tapas-large-finetuned-wtq',
                        help='model to use. Accepted models are [tapas, tapex, chatgpt]')
    parser.add_argument('--db_category', type=str, default='ecommerce',
                        help='the proprietary category used to generate the tests. '
                             'Accepted categories are [ecommerce, finance, medicine, miscellaneous]')
    parser.add_argument('--tbl_names', default=None, nargs='*',
                        help='the proprietary tables in the selected category. '
                             'Only used if you want to test a specific table.')
    parser.add_argument('--metrics', default=None, nargs='*',
                        help='the the metrics to use for evaluation. '
                             'Accepted metrics are [cell_precision, cell_recall, tuple_constraint,'
                             ' tuple_cardinality, tuple_order]')
    parser.add_argument('--sql_generators', default=None, nargs='*',
                        help='the sql generators to use for test generation. '
                             'Accepted generators are [select, orderby, distinct, where]')
    parser.add_argument('--input_base_path', type=str, default='data',
                        help='the base path where the proprietary data is stored. '
                             'Default is data')
    parser.add_argument('--output_db_path', type=str, default='output',
                        help='the path where the generated tests will be stored. '
                             'Default is output')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    main(parser.parse_args())
