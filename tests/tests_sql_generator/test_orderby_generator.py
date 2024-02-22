from unittest.mock import Mock

import pytest

from qatch.sql_generator import OrderByGenerator


@pytest.fixture
def mock_database():
    mock_db = Mock()
    mock_db.get_schema_given.return_value = Mock(name='schema', spec=['name'])
    mock_db.get_schema_given.return_value.name.tolist.return_value = ['col1', 'col2', 'col3']
    return mock_db


def test_generate_order_asc(mock_database):
    generator = OrderByGenerator(mock_database)
    table_name = 'example_table'
    generator._generate_order_asc(table_name, ['col1', 'col2', 'col3'])
    assert 'ORDERBY-SINGLE' in generator.sql_generated['sql_tags']
    query = f'SELECT * FROM `{table_name}` ORDER BY `col1` ASC'
    question = f'Show all data ordered by "col1" in ascending order for the table "{table_name}"'

    assert query in generator.sql_generated['queries']
    assert question in generator.sql_generated['questions']


def test_generate_order_desc(mock_database):
    generator = OrderByGenerator(mock_database)
    table_name = 'example_table'
    generator._generate_order_desc(table_name, ['col1', 'col2', 'col3'])
    assert 'ORDERBY-SINGLE' in generator.sql_generated['sql_tags']
    query = f'SELECT * FROM `{table_name}` ORDER BY `col1` DESC'
    question = f'Show all data ordered by col1 in descending order for the table {table_name}'

    assert query in generator.sql_generated['queries']
    assert question in generator.sql_generated['questions']
