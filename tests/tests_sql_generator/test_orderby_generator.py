from unittest.mock import Mock

import pytest

from qatch.sql_generator import OrderByGenerator


@pytest.fixture
def mock_database():
    mock_db = Mock()
    mock_db.get_schema_given.return_value = Mock(name='schema', spec=['name'])
    mock_db.get_schema_given.return_value.name.tolist.return_value = ['col1', 'col2', 'col3']
    return mock_db


def test_order_by_generator(mock_database):
    generator = OrderByGenerator(mock_database)
    table_name = 'example_table'

    generated_sql = generator.sql_generate(table_name)

    assert 'ORDERBY-SINGLE' in generated_sql['sql_tags']

    assert len(generated_sql['queries']) == 6  # three ascending, three descending

    query = f'SELECT * FROM "{table_name}" ORDER BY "col1" ASC'
    question = f'Show all data ordered by "col1" in ascending order for the table "{table_name}"'

    assert query in generated_sql['queries']
    assert question in generated_sql['questions']
