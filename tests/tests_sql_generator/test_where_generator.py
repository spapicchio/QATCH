from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from qatch.sql_generator import WhereGenerator


class TestWhereGenerator:

    @pytest.fixture
    def sample_data(self):
        # Provide sample data for testing
        # You can customize this fixture based on your requirements
        data = {
            'column1': [1, 2, 3, 4],
            'column2': ['A', 'B', 'A', 'C'],
            'column3': [10.5, 20.3, 15.2, 18.7]
        }
        df = pd.DataFrame(data)
        return df

    @pytest.fixture
    def mock_single_database(self, sample_data):
        single_database = Mock()
        single_database.get_schema_given.return_value = sample_data.columns.tolist()
        single_database.get_table_given.return_value = sample_data
        return single_database

    @pytest.fixture
    def where_generator(self, mock_single_database):
        return WhereGenerator(mock_single_database)

    def test_generate_where_categorical(self, sample_data, where_generator):
        # Test case for generating WHERE queries and questions for categorical columns
        table_name = 'sample_table'
        cat_cols = ['column2']
        where_generator._generate_where_categorical(table_name, cat_cols, sample_data)

        # Perform assertions on the generated SQL tags, queries, and questions
        assert where_generator.sql_generated['sql_tags'] == ['WHERE-CAT-MOST-FREQUENT', 'WHERE-CAT-LEAST-FREQUENT',
                                                             'WHERE-CAT-MOST-FREQUENT', 'WHERE-CAT-LEAST-FREQUENT',
                                                             'WHERE-NOT-MOST-FREQUENT', 'WHERE-NOT-LEAST-FREQUENT']
        assert len(where_generator.sql_generated['queries']) == 6
        assert len(where_generator.sql_generated['questions']) == 6

    def test_generate_where_numerical(self, sample_data, where_generator):
        # Test case for generating WHERE queries and questions for numerical columns
        table_name = 'sample_table'
        num_cols = ['column1', 'column3']
        where_generator._generate_where_numerical(table_name, num_cols, sample_data)

        # generate 6 queries (max, min, mean * 2 [>, <]) for each column,
        # so in this case we expect 12 queries
        assert len(where_generator.sql_generated['sql_tags']) == 12
        assert len(where_generator.sql_generated['queries']) == 12
        assert len(where_generator.sql_generated['questions']) == 12

        assert 'SELECT * FROM "sample_table" WHERE "column1" > "4"' in where_generator.sql_generated['queries']
        assert 'SELECT * FROM "sample_table" WHERE "column1" < "4"' in where_generator.sql_generated['queries']
        assert 'Show the data of the table "sample_table" where "column1" is greater than 4' in \
               where_generator.sql_generated['questions']
        assert 'Show the data of the table "sample_table" where "column1" is less than 4' in \
               where_generator.sql_generated['questions']

    def test_get_most_frequent_or_max_value(self, where_generator):
        # Test case with numerical values
        num_values = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        result = where_generator._get_most_frequent_or_max_value(num_values)
        assert result == 4

        # Test case with categorical values (most frequent value)
        cat_values = np.array(['A', 'B', 'A', 'C', 'C', 'C', 'D', 'D', 'D', 'D'])
        result = where_generator._get_most_frequent_or_max_value(cat_values)
        assert result == 'D'

        # Test case with categorical values and Null (most frequent value
        cat_values = np.array(['A', None, 'A', 'C', None, 'C', 'D', 'D', 'D', 'D'])
        result = where_generator._get_most_frequent_or_max_value(cat_values)
        assert result == 'D'
        # Test case with empty array
        empty_values = np.array([])
        result = where_generator._get_most_frequent_or_max_value(empty_values)
        assert result is None

        # Test case with null values
        null_values = np.array([1, 2, None, 3, 3, 3, 4, 4, 4, 4])
        result = where_generator._get_most_frequent_or_max_value(null_values)
        assert result == 4

    def test_get_least_frequent_or_min_value(self, where_generator):
        # Test case with numerical values
        num_values = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        result = where_generator._get_least_frequent_or_min_value(num_values)
        assert result == 1

        # Test case with categorical values (most frequent value)
        cat_values = np.array(['A', 'B', 'A', 'C', 'C', 'C', 'D', 'D', 'D', 'D'])
        result = where_generator._get_least_frequent_or_min_value(cat_values)
        assert result == 'B'

        # Test case with categorical values and Null (most frequent value
        cat_values = np.array(['A', None, 'A', 'C', None, 'C', 'D', 'D', 'D', 'D'])
        result = where_generator._get_least_frequent_or_min_value(cat_values)
        assert result == 'A'
        # Test case with empty array
        empty_values = np.array([])
        result = where_generator._get_least_frequent_or_min_value(empty_values)
        assert result is None

        # Test case with null values
        null_values = np.array([1, 2, None, 3, 3, 3, 4, 4, 4, 4])
        result = where_generator._get_least_frequent_or_min_value(null_values)
        assert result == 1

    def test_get_median_value(self, where_generator):
        num_values = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        result = where_generator._get_median_value(num_values)
        assert result == 3
