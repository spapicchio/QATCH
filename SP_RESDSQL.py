import argparse
import logging

from utils import init_metric_evaluator, get_summary, \
    init_test_generator, get_predictions_results_from_dbs


def main(args):
    if args.verbose:
        params_dict = vars(args)
        logging.basicConfig(level=logging.INFO)
        logging.info('Script arguments:')
        [logging.info(f'\t{key}: {value}') for key, value in params_dict.items()]

    # init QATCH components
    test_generator = init_test_generator(args.db_category,
                                         'sp',
                                         args.output_db_path,
                                         args.input_base_path)
    metric_evaluator = init_metric_evaluator(args.metrics)

    # 1. QATCH-Generate
    if args.verbose:
        logging.info('Starting generating tests')
    tables, tests_df = test_generator.generate(generators=args.sql_generators,
                                               table_names=args.tbl_names,
                                               save_spider_format=True)

    # 2. TRL model predictions
    """
    This step s outside of QATCH and can be performed following
    the official github link of the model (https://github.com/RUCKBReasoning/RESDSQL)
    We use the script infer_text2sql.sh with the large t5 model.
    """

    # 3. QATCH-Evaluate
    """
    - For semantic-parsing i.e. text-to-sql the output of the model is a query. 
    - However, The QATCH-Evaluation tags are based on the returned table
        rather than the query itself.
    - We provide the method get_predictions_results_from_dbs to run the query
      over the input databases.
    """
    tests_df = get_predictions_results_from_dbs(base_path_db=args.output_db_path,
                                                df=tests_df,
                                                predictions='query')
    if args.verbose:
        logging.info('Starting evaluating tests')
    tests_df = metric_evaluator.evaluate_with_df(tests_df,
                                                 target='query_result',
                                                 predictions='query_result_predictions')
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
    parser = argparse.ArgumentParser(prog='SP_RESDSQL',
                                     description='Semantic Parsing for proprietary data')
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
    parser.add_argument('--output_db_path', type=str, default='output_SP',
                        help='the path where the generated tests will be stored. '
                             'Default is output_SP')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    main(parser.parse_args())
