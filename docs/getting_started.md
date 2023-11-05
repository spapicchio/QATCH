

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
However, in case you want to use our implementation you can add the extras requirements.

```console
# Using poetry (recommended)
poetry add QATCH -E model

# Using pip
pip install QATCH[model] 
```

## Create connection with input data

Once you have installed QATCH, you need to create a connection between your data and the tool.
If your data is not stored in a sqlite database you can use our code to generate it.
If this is not the case, you can skip this passage.

```python
from qatch.database_reader import SingleDatabase
import pandas as pd

# Create dummy table
data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)

# define the tables in the database (<table_name> : <table>)
db_tables = {'olympic_games': table}

# define where to store the sqlite database
db_save_path = 'test_db'

# define the name of the database
db_id = 'olympic'

# create database connection
db = SingleDatabase(db_path=db_save_path, db_name=db_id, tables=db_tables)
```

This class will create the sqlite database in "db_save_path/db_id/db_id.sqlite".

Once you have the database stored in this format "db_save_path/db_id/db_id.sqlite",
you can create a connection in the following way:

```python
from qatch.database_reader import MultipleDatabases

# The path to multiple databases
db_save_path = 'test_db'
databases = MultipleDatabases(db_save_path)
```

This class automatically detects the available databases and handle the communication
between the code and the sqlite files.

## Step 1: QATCH generate

```python
from qatch import TestGenerator

# init generator
test_generator = TestGenerator(databases=databases)

# generate tests for each database and for each generator
tests_df = test_generator.generate()
```

Test generator automatically creates a checklist based on the proprietary data.
The tests_df dataframe contains:

- *db_id*: The database name associated with the test.
- *tbl_name*: The table name associated with the test.
- *sql_tags*: the SQL generator used to create the test.
- *query*: The generated query. Used to evaluate the model.
- *question*: The generated question associated with the query. Used as input for the model.


## Step 2: TRL model predictions

QATCH is intended to be used without the inference step.
However, it supports several models for reproducibility reason.

```python
from tqdm import tqdm

from qatch.models import Tapas

# init the model 
model = Tapas(model_name=google / tapas - large - finetuned - wtq)

# iterate for each row and run prediction
tqdm.pandas(desc=f'Predicting for {model.name}')
tests_df[f'predictions_{model.name}'] = tests_df.progress_apply(
    lambda row: model.predict(
        table=databases.get_table(db_id=row['db_id'], tbl_name=row['tbl_name']),
        query=row['question'],
        tbl_name=row['tbl_name']
    ),
    axis=1
)
```

Since Tapas, Tapex, Omnitab and LLama2 are based on huggingFace, the model_name parameter can be
any possible name associate with the model in the hub.

To use ChatGPT_QA or ChatGPT_SP you need to provide the API credentials:

```python
from qatch.models import ChatGPT_QA

model = ChatGPT_QA(model_name="gpt-3.5-turbo-0613",
                   api_key="your_api_key_chatgpt",
                   api_org="your_api_org_chatgpt")
```

To use LLama2_QA or LLama2_SP you need to specify the HuggingFace token

```python
from qatch.models import LLama2_QA

model = LLama2_QA(model_name="meta-llama/Llama-2-7b-chat-hf",
                  hugging_face_token="your_hugging_face_token")
```
The tests_df dataframe after the prediction phase contains:

- *db_id*: The database name associated with the test.
- *tbl_name*: The table name associated with the test.
- *sql_tags*: the SQL generator used to create the test.
- *query*: The generated query. Used to evaluate the model.
- *question*: The generated question associated with the query. Used as input for the model.
- *predictions_<model_used>*: The predicted query/cells based on the task (SP or QA respectively)

## Step 3: QATCH evaluate

QATCH MetricEvaluator is composed of 5 metrics (3 intra-tuple and 2 inter-tuple).

```python
from qatch import MetricEvaluator

evaluator = MetricEvaluator(databases=databases)
tests_df = evaluator.evaluate_with_df(tests_df,
                                      prediction_col_name="<prediction_col_name>",
                                      task="QA")
```
The final dataframe contains:

- *db_id*: The database name associated with the test.
- *tbl_name*: The table name associated with the test.
- *sql_tags*: the SQL generator associated with the test.
- *query*: The generated query from step 1.
- *question*: The generated question from step 1. Used as input for the model.
- *predictions_<model_used>*: The predicted query/cells from step 2.
- *5 metrics*: The metrics used to evaluate the models.
