from unittest.mock import Mock, MagicMock

import pytest

from test_generator.sql_generator import OrderByGenerator


class TestOrderByGenerator:
    @pytest.fixture
    def database(self):
        database = Mock()
        database.get_columns_from_table.return_value = ['Name', 'Surname']
        database.run_query.side_effect = MagicMock(return_value='result')
        return database

    @pytest.fixture
    def generator(self, database):
        # Create an instance of OrderByGenerator with the mock database
        return OrderByGenerator(database)

    def test_sql_generate_len(self, generator):
        generate_dict = generator.sql_generate('table_name')
        assert len(generate_dict.keys()) == 4
        assert len(generate_dict['queries']) == len(generate_dict['questions']) == \
               len(generate_dict['sql_tags']) == len(generate_dict['results'])

    def test_sql_generate_called_twice(self, generator):
        generate_dict_1 = generator.sql_generate('table_name')
        generate_dict_2 = generator.sql_generate('table_name')
        assert len(generate_dict_1['queries']) == len(generate_dict_2['queries'])
        assert len(generate_dict_1['questions']) == len(generate_dict_2['questions'])
        assert len(generate_dict_1['sql_tags']) == len(generate_dict_2['sql_tags'])
        assert len(generate_dict_1['results']) == len(generate_dict_2['results'])

    def test_generate_order_asc(self, generator):
        # Call the method to be tested
        generate_dict = generator.generate_order_asc('table_name', ['Name', 'Surname'])

        target_queries = ['SELECT * FROM "table_name" ORDER BY "Name" ASC',
                          'SELECT * FROM "table_name" ORDER BY "Surname" ASC']
        assert target_queries == generate_dict['queries']
        target_questions = [
            'Show all data ordered by "Name" in ascending order for the table "table_name"',
            'Show all data ordered by "Surname" in ascending order for the table "table_name"',
        ]
        assert target_questions == generate_dict['questions']
        assert len(generate_dict['queries']) == len(generate_dict['questions']) == \
               len(generate_dict['sql_tags']) == len(generate_dict['results'])

    def test_generate_order_desc(self, generator):
        # Call the method to be tested
        generate_dict = generator.generate_order_desc('table_name', ['Name', 'Surname'])

        target_queries = ['SELECT * FROM "table_name" ORDER BY "Name" DESC',
                          'SELECT * FROM "table_name" ORDER BY "Surname" DESC']
        assert target_queries == generate_dict['queries']
        target_questions = [
            'Show all data ordered by "Name" in descending order for the table "table_name"',
            'Show all data ordered by "Surname" in descending order for the table "table_name"',
        ]
        assert target_questions == generate_dict['questions']

        assert len(generate_dict['queries']) == len(generate_dict['questions']) == \
               len(generate_dict['sql_tags']) == len(generate_dict['results'])
