import argparse
import logging

from tqdm import tqdm

from utils import init_model, init_metric_evaluator, init_test_generator_spider, get_summary


def main(args):
    if args.verbose:
        params_dict = vars(args)
        logging.basicConfig(level=logging.INFO)
        logging.info('Script arguments:')
        [logging.info(f'\t{key}: {value}') for key, value in params_dict.items()]

    # init QATCH components
    model = init_model(args.model_name)
    test_generator = init_test_generator_spider(args.spider_base_path)
    metric_evaluator = init_metric_evaluator(args.metrics)

    # 1. QATCH-Generate
    if args.verbose:
        logging.info('Starting generating tests')
    tables, tests_df = test_generator.generate(cat_granularity=args.cat_granularity,
                                               sql_granularity=args.sql_granularity)

    # 2. TRL model predictions
    if args.verbose:
        logging.info('Starting predictions')
    tqdm.pandas(desc='predictions')
    tests_df['predictions'] = tests_df.progress_apply(
        lambda row: model.predict(table=tables[(row.db_id, row.tbl_name)], queries=row.question),
        axis=1
    )

    # 3. QATCH-Evaluate
    if args.verbose:
        logging.info('Starting evaluating tests')
    tests_df = metric_evaluator.evaluate_with_df(tests_df,
                                                 target='query_result',
                                                 predictions='predictions')
    # 4. save tests results and summary
    if args.verbose:
        logging.info('Save results')
    path = f'{args.output_path}/tests_with_results_spider.json'
    tests_df.reset_index(drop=True).to_json(path)
    if args.verbose:
        logging.info(f'Saved tests in {path}')
    results = get_summary(tests_df, args.metrics)
    path = f'{args.output_path}/summary_spider.csv'
    results.to_csv(path)
    if args.verbose:
        logging.info(f'Saved summary in {path}')


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(prog='QA_SPIDER',
                                     description='Run question answering for SPIDER')

    parser.add_argument('--model_name', type=str, default='google/tapas-large-finetuned-wtq',
                        help='model to use. Accepted models are [tapas, tapex, chatgpt]')

    parser.add_argument('--spider_base_path', type=str, default='data/spider',
                        help='where the spider dataset is stored, default is data/spider')

    parser.add_argument('--metrics', default=None, nargs='*',
                        help='the the metrics to use for evaluation. '
                             'Accepted metrics are [cell_precision, cell_recall, tuple_constraint,'
                             ' tuple_cardinality, tuple_order]')

    parser.add_argument('--sql_granularity', default='select', nargs='*',
                        help='the sql generators to use for test generation. '
                             'Accepted generators are [SELECT, ORDERBY, SIMPLE_AGGR, '
                             'WHERE, GROUPBY, HAVING]. \n'
                             'If None is passed, all of them are generated')

    parser.add_argument('--cat_granularity', default='low', nargs='*',
                        help='the cat generators to use for test generation. '
                             'Accepted generators are [LOW, MEDIUM, HIGH]\n'
                             'If None is passed, all of them are generated')

    parser.add_argument('--output_path', type=str, default='output',
                        help='the path where the generated tests will be stored. '
                             'Default is output')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    main(parser.parse_args())
