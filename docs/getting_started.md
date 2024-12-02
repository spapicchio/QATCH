## Installation

First need to install QATCH.
You can do this by running the following command:

```console
# Using poetry (recommended)
poetry add QATCH

# Using pip
pip install QATCH 
```

Since QATCH is intended to be used without the inference step, the base installation does not come
with the models' requirements.

## QATCH package

The toolbox is composed of three main packages:

- connectors: Connects the database to the base code
- generate_dataset: Contains the code to create the dataset
- evaluate_dataset: Contains the code to evaluate the dataset

## Create connection with input data

Once you have installed QATCH, you need to create a connection between your data and the tool.
If your data is not stored in a sqlite database you can use our code to generate it.
If this is not the case, you can skip this passage.

```python
import pandas as pd

from qatch.connectors.sqlite_connector import SqliteConnector

# Create dummy table
data = {
    "id": [0, 1, 2, 3, 4, 5],
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)

# define the tables in the database (<table_name> : <table>)
db_tables = {'olympic_games': table}

# Assume the PKs have all different names. Two tables cannot have same PK name.
table2primary_key = {'olympic_games': 'id'}

# define where to store the sqlite database
db_save_path = 'test_db.sqlite'

# define the name of the database
db_id = 'olympic'

# create database connection
connector = SqliteConnector(
    relative_db_path=db_save_path,
    db_name=db_id,
    tables=db_tables,
    table2primary_key=table2primary_key
)
```

This class will create the sqlite database in db_save_path.

If you want to directly connect to the sqlite database:

```python
from qatch.connectors.sqlite_connector import SqliteConnector

db_save_path = 'test_db.sqlite'
db_name = 'olympics'
connector = SqliteConnector(
    relative_db_path=db_save_path,
    db_name=db_name,
)
```

## Step 1: QATCH generate

To generate the datasets, we need an orchestrator:

```python
from qatch.connectors.sqlite_connector import SqliteConnector
from qatch.generate_dataset.orchestrator_generator import OrchestratorGenerator

# connection to the database
connector = SqliteConnector(
    relative_db_path='<your_sqlite_path>',
    db_name='<your_db_name>',
)

# init the orchestrator
orchestrator_generator = OrchestratorGenerator()

# test generation
orchestrator_generator.generate_dataset(connector)
```

Test generator automatically creates a checklist based on the proprietary data.
The tests_df dataframe contains:

- *db_path*: The database path associated with the test
- *db_id*: The database name associated with the test.
- *tbl_name*: The table name associated with the test.
- *test_category*: The test category.
- *sql_tag*: A more granular label for the test category.
- *query*: The generated query. Used to evaluate the model.
- *question*: The generated question associated with the query. Used as input for the model.

## Step 2: TRL model predictions

QATCH is intended to be used without the inference step. the new release of QATCH deprecate this section. 
For reproducibility purposes, refer to previous main version of QATCH starting with 0.* 

## Step 3: QATCH evaluate

Supported metrics are:
- Cell Precision: [0-1] how many predicted elements are in target
- Cell Recall: [0-1] how many target elements are in prediction
- Tuple Cardinality: [0-1] whether cardinality of target and prediction matches
- Tuple Constraint: [0-1] whether the tuple constraint is respected or not 
- Tuple Order: [0-1] whether prediction and target contains same order, calculated only for target query with ORDER-BY clause
- Execution Accuracy: [0-1] whether the execution of the query is the same or not.

There are two options to evaluate your predictions: With a DataFrame or with a single test.

```python
from qatch.connectors.sqlite_connector import SqliteConnector
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator

# init orchestrator evaluator 
evaluator = OrchestratorEvaluator()

connector = SqliteConnector(
    relative_db_path='<your_sqlite_path>',
    db_name='<your_db_name>',
)

# solution with df:
# Returns: The input dataframe enriched with the metrics computed for each test case.
evaluator.evaluate_df(
    df='<the pandas df>',
    target_col_name='<target_column_name>',
    prediction_col_name='<prediction_column_name>',
    db_path_name='<sqlite_db_path>'
)

# Returns: A dictionary comprising the evaluation metrics values for the test.
evaluator.evaluate_single_test(
    target_query='SELECT * FROM T',
    predicted_query='SELECT * FROM T',
    connector=connector
)

# note that target and prediction can be interchangeable the execution of the query or the SQL query
#  The result is the same
evaluator.evaluate_single_test(
    target_query=[[1, 2], [3, 4]],
    predicted_query=[[1, 2], [3, 4]],
    connector=connector
)

```