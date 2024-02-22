import json

import pandas as pd

from qatch import TestGenerator
from qatch.database_reader import SingleDatabase, MultipleDatabases
from utils import create_json_tables_file
from .inject_ambiguity import inject_ambiguity


def damber_generate_ambiguous_tests(db_save_path: str,
                                    input_path_tables: str = 'damber/data/ambiguous_tables.xlsx',
                                    ambiguity_annotation_file: str = 'damber/data/ambiguity_annotation_task_1.xlsx',
                                    ambiguity_th: float = 2.0):
    """generate ambiguous tests from databases specified in the input path, and save to given save path."""
    # Load the specified databases and store them in the variable 'databases'
    # This function _load_databases might involve reading the file at 'input_path', parsing it into a suitable format,
    # and then storing the databases into 'db_save_path'. This might involve creating connections to these databases
    databases = _load_databases(input_path_tables, db_save_path)

    # Generate tests from the loaded databases
    # The tests generation process might involve executing some SQL queries or performing some database operations.
    # The specific details would depend on the implementation of '_generate_tests'
    tests_df = _generate_tests(databases)

    # Filter/optimize the generated tests. Perhaps this step cleans up the tests or selects them based on some criteria
    # The function _filter_tests does this and it returns a new DataFrame possibly where each row is a test
    tests_df = _filter_tests(tests_df)

    # inject ambiguity
    # the df now will contain new columns
    # ['ambiguous_questions', 'swapped_attribute', 'ambiguous_labels', 'ambiguous_attributes']
    tests_df = inject_ambiguity(tests_df, ambiguity_annotation_file, ambiguity_th)
    # Save the filtered tests into the path 'db_save_path'.
    _save_tests(tests_df, db_save_path)

    # Save the database tables into the same path 'db_save_path'
    _save_tables(db_save_path)

    # Return the DataFrame containing the filtered tests
    return tests_df


def _load_databases(input_path: str, db_save_path: str):
    """Load database tables and create a connection to MultipleDatabase object."""
    db_tables = pd.read_excel(input_path, None)

    for db_id in db_tables:
        # Create database connection
        _ = SingleDatabase(
            db_path=db_save_path,
            db_name=db_id.replace('-', '_'),
            tables={db_id.replace('-', '_'): db_tables[db_id]}
        )

    databases = MultipleDatabases(db_save_path)

    return databases


def _generate_tests(databases):
    """
    Initialize TestGenerator on the given databases and generate tests.
    """
    # Init generator
    test_generator = TestGenerator(databases=databases)

    # Generate tests for each database and for each generator
    tests_df = test_generator.generate(
        generators=None,
        db_names=None,
        seed=2023
    )

    return tests_df


def _filter_tests(tests_df):
    """Filter out unnecessary for this analysis SQL tags in provided tests DataFrame."""
    tags_to_remove = [
        'HAVING-AGG-SUM-LS',
        'HAVING-AGG-AVG-LS',
        'DISTINCT-MULT',
        'WHERE-CAT-LEAST-FREQUENT',
        'WHERE-NOT-LEAST-FREQUENT',
        'WHERE-NOT-MOST-FREQUENT',
        'HAVING-COUNT-LS',
        'HAVING-COUNT-GR',
        'WHERE-NUM-MIN-VALUES',
        'WHERE-NUM-MAX-VALUES',
        'HAVING-AGG-AVG-GR'
    ]

    mask = tests_df['sql_tags'].isin(tags_to_remove)
    tests_df = tests_df[~mask]

    return tests_df


def _save_tests(tests_df, db_save_path):
    """Save test DataFrame to a JSON test file."""
    tests_df.to_json(f'{db_save_path}/ambiguous_tests_df.json', orient='records')


def _save_tables(db_save_path):
    """Save tables to a JSON table file."""
    tables = create_json_tables_file(db_save_path)

    with open(f'{db_save_path}/ambiguous_tables.json', 'w') as outfile:
        json.dump(tables, outfile, sort_keys=True, indent=2, separators=(',', ': '))
