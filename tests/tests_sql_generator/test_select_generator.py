from unittest.mock import Mock

import pytest

from qatch.sql_generator.select_generator import SelectGenerator


class TestSelectGenerator:
    @pytest.fixture
    def mock_single_database(self):
        single_database = Mock()
        single_database.get_schema_given.return_value = Mock(name='schema', spec=['name'])
        single_database.get_schema_given.return_value.name.tolist.return_value = ['col1', 'col2', 'col3']
        return single_database

    @pytest.fixture
    def select_generator(self, mock_single_database):
        return SelectGenerator(mock_single_database)

    def test_select_all_table(self, select_generator, mock_single_database):
        select_generator._select_all_table('test_table')
        assert select_generator.sql_generated == {
            'sql_tags': ['SELECT-ALL'],
            'queries': ['SELECT * FROM "test_table"'],
            'questions': ['Show all the rows in the table test_table']
        }

    def test_select_add_col(self, select_generator, mock_single_database):
        select_generator._select_add_col('test_table')
        assert select_generator.sql_generated == {
            'sql_tags': ['SELECT-ADD-COL', 'SELECT-ADD-COL'],
            'queries': ['SELECT "col1" FROM "test_table"',
                        'SELECT "col1", "col2" FROM "test_table"'],
            'questions': ['Show all "col1" in the table test_table',
                          'Show all "col1", "col2" in the table test_table'],
        }

    def test_select_random_col(self, select_generator, mock_single_database):
        select_generator._select_random_col('test_table')

        assert select_generator.sql_generated == {
            'sql_tags': ['SELECT-RANDOM-COL', 'SELECT-RANDOM-COL', 'SELECT-RANDOM-COL'],
            'queries': ['SELECT "col2" FROM "test_table"',
                        'SELECT "col3", "col2" FROM "test_table"',
                        'SELECT "col2", "col3", "col1" FROM "test_table"'],
            'questions': ['Show all "col2" in the table test_table',
                          'Show all "col3", "col2" in the table test_table',
                          'Show all "col2", "col3", "col1" in the table test_table']
        }
