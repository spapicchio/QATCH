import argparse

from runners import RunnerProprietary


def main(args):
    # initialize the runner
    runner = RunnerProprietary(**vars(args))
    # generate tests
    tables, tests_df = runner.generate_tests()
    # predict
    tests_df = runner.predict(tables, tests_df)
    # evaluate the predictions
    tests_df = runner.evaluate(tests_df)


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(prog='', description='')

    parser.add_argument('--model_name_path', type=str, default='chatgpt',
                        help='model to use. Default is google/tapas-large-finetuned-wtq')

    parser.add_argument('--db_id', type=str, default='ecommerce',
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

    parser.add_argument('--input_path_tables', type=str, default='data',
                        help='the base path where the proprietary data is stored. '
                             'Default is data')

    parser.add_argument('--save_database_tests', type=str, default='temp',
                        help='the path where the generated tests will be stored. '
                             'Default is output')

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    parser.add_argument("--percentage", type=float, default=0.0,
                        help='TODO')  # TODO

    parser.add_argument("--seed", type=float, default=2023,
                        help='TODO')  # TODO

    parser.add_argument("-spf", "--save_spider_format", action="store_false",
                        help='TODO')  # TODO

    parser.add_argument("--task", type=str, default='SP',
                        help='TODO')  # TODO

    main(parser.parse_args())
