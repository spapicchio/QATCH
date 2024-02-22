# QATCH: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data

<p align="center">
 <kbd>
  <img src="docs/img/qatch_logo_verticale.jpg" alt="QATCH's logo" height="300" style="border-radius:50%">
 </kbd>
 </p>

This repository is the official implementation of [QATCH: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data](https://openreview.net/forum?id=XOpaPrb0U5)
to appear in NeurIPS Dataset and Benchmark track 2023.

# 🔥 Updates
- [**2024-Jan-22**]: Add [DAMBER: (Data-AMBiguity testER)](https://github.com/spapicchio/QATCH/tree/master/damber#readme) 
- [**2024-Jan-10**]: Add JOIN tests for proprietary data 
- [**2023-Dec-15**]: new License: Apache-2.0 
- [**2023-Nov-06**]: Camera ready version is now available! [check it out](https://openreview.net/forum?id=XOpaPrb0U5)! 
- [**2023-Nov-05**]: QATCH can now be donwloaded from pip! Do not forget to check the [documentation](https://spapicchio.github.io/QATCH/)! 

# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SNoy3GZGPWltVS5cL068xAG9YoPS_3_l?usp=sharing)

* ***What is QATCH?*** **Q**uery-**A**ided **T**RL **Ch**ecklist (QATCH) is a toolbox to highlight TRL models’ strengths
  and weaknesses on prorietary tables for Question Answering (QA) and Semantic Parsing (SP).
* ***How does it work?*** Given a proprietary database as input, it generates a testing checklist for QA and SP.
* ***More specifically?*** A query generation algorithm crafts tests by means of the expressive power of SQL.
* ***Ok cool, that's it?*** To evaluate the model's predictions, we propose 5 new metrics intra and inter tuple.
* ***Where is processed the data?*** The data is processed locally. We do not store any data. If you use the ChatGPT
  wrapper the data is processed by OpenAI.
* ***Where can I check the results?*** The generated tests along with the predictions and the metric scores can be
  downloaded [here](https://drive.google.com/uc?export=download&id=1_z8N52QNAHnxpHv54VhbvYu7DbKV6QRv). This is to
  prevent the costly generation of test results with the openAI API and to build trust in our results.


 <figure style="text-align:center">
  <img src="docs/img/qatch-full-pipeline.png">
</figure>

<br>

QATCH's automatically generates and evaluates test checklists on TRL models based on the three-step process depicted
below:

1. *QATCH-Generate*. It generates a set of queries tailored to proprietary data. For each query it formulates both the
   SQL declaration, its free-text version, and the expected ground truth consisting of table instances.
   The SQL declaration expresses the logical complexity of the query and reflects the presence/absence of specific
   features peculiar to relational data model such as presence of missing values and duplicate values.

2. *TRL Model Predictions*. It processes the tests for various TRL models and tasks. The current toolbox version
   supports three Table Representation Learning (TRL) models for
   QA: [TAPAS](https://github.com/google-research/tapas), [TAPEX](https://github.com/microsoft/Table-Pretraining)
   and [Omnitab](https://github.com/jzbjyb/OmniTab).
   In addition, two LLMs are implemented for QA and SP [ChatGPT 3.5](https://openai.com/blog/chatgpt) (need the API key)
   and [LLama2](https://huggingface.co/blog/llama2) (need the HuggingFace token).

3. *QATCH-Evaluate*. It evaluates the models outputs according to a set of cross-task performance metrics.

<p align="center">
<img src="docs/img/measures.png" width="70%">
</p>

QATCH’s metrics are computed between the model output (prediction) and expected 
ground-truth results (target). The target is the answer of the NL question "Show me all the data" over
a table with three tuples and two attributes.
<br>

Given the ground truth result (target) with three tuples over two attributes, we report the metric values for five
predictions, coming either from a QA or from the execution of a query in SP. More details can be found in
the [metrics](qatch/metrics) folder

## Who should use QATCH?

QATCH is designed to create "behavioral testing" checklist for QA and SP tasks.
The checklist is used to understand in which case the models fail when processing proprietary data for QA and SP tasks.

In a corporate setting, there are at least three scenarios where a given TRL model needs to be evaluated
against proprietary datasets:

- Comparison: Compare TRL models fine-tuned on private examples to see which one performs best.
- Validation: As crafting examples is expensive, verify when the quality meets the requirements.
- Maintenance: Fine-tuned models need to be re-calibrated to avoid data and conceptual shifting,
  continuous evaluation helps the identification of this issue.

But the usage of QATCH it is not limited to the TRL models. Indeed, we propose two scenarios
where QATCH can be used with LLMs:

- LLM compatibility version: Compare different version of the same LLMs to see the best performing one.
- Prompt engineering: Analyse the best prompt definition based on the proprietary data.

<p align="center">
<img src="docs/img/use_case_walter.png" width="70%">
</p>

Use case example of engineer Walter. 
With QATCH it is able to create a model ranking on his proprietary data for QA and SP.


## Project

```shell
|-- metric_evaluator.py # user interface to calculate metrics for QA or SP
|-- test_generator.py # user interface to run different SQL generators
|--database_reader
    |-- single_database.py # initialise single database 
    |-- multiple_databases.py # handle multiple single database instances
|-- metrics
    |-- metric_evaluator.py # wrapper to initialise the user selected metrics
    |-- abstract_metric.py # abstract class to handle common metric methods
    |-- cell_precision_tag.py # implement cell precision tag
    |-- cell_recall_tag.py # implement cell recall tag
    |-- tuple_cardinality_tag.py # implement tuple cardinality tag
    |-- tuple_constraint_tag.py # implement tuple constraint tag
    |-- tuple_order_tag.py # implement tuple order tag
|-- models
    |-- chatgpt
        |-- abstract_chatgpt.py # abstract class to handle common methods for ChatGPT
        |-- chatgpt_QA.py # implement chatgpt for QA task
        |-- chatgpt_SP.py # implement chatgpt for SP task
    |-- chatgpt
        |-- abstract_llama2.py # abstract class to handle common methods for LLama2
        |-- llama2_QA.py # implement llama2 for QA task
        |-- llama2_SP.py # implement llama2 for SP task
    |-- abstract_model.py # abstract class to handle common model methods
    |-- tapas.py # implement input processing and the prediction for TAPAS
    |-- tapex.py # implement input processing and the prediction for TAPEX
    |-- omnitab.py # implement the input processing and the prediction for Omnitab
|-- sql_generator
    |-- abstract_sql_generator.py # handle common methods for SQL generators
    |-- select_generator.py # implement SELECT tests
    |-- distinct_generator.py # implement DISTINCT tests
    |-- orderby_generator.py # implement ORDERBY tests
    |-- where_generator.py # implement WHERE tests
    |-- groupby_generator.py # implement GROUPBY tests
    |-- having_generator.py # implement HAVING tests
    |-- simple_agg_generator.py # implement SIMPLE AGG tests
    |-- null_generator.py # implement NULL generator tests

```

# ⚡️ Quickstart
## Installation
You can install QATCH by running the following commands:

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

## How to use QATCH with my data?

1. Load your input data

Create a connection between your data and the tool.
If your data is not stored in a sqlite database you can use our code to generate it.

```python
import pandas as pd
from qatch.database_reader import SingleDatabase

data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)
db_tables = {'olympic_games': table}

# create database connection
# create the sqlite database in "db_save_path/db_id/db_id.sqlite".
db = SingleDatabase(db_path="db_save_path", db_name="db_id", tables=db_tables)
```
Now we can create a connection with multiple databases:

```python
from qatch.database_reader import MultipleDatabases

# The path to multiple databases
db_save_path = 'test_db'
databases = MultipleDatabases(db_save_path)
```

2. QATCH-Generate: Generates the tests

```python
from qatch import TestGenerator

# init generator
test_generator = TestGenerator(databases=databases)

# generate tests for each database and for each generator
tests_df = test_generator.generate()
```

3. TRL Model Predictions: if you want to use any version of Tapas/Tapex for QA in Huggingface or chatGPT you can use the
   already implemented modules but it is NOT mandatory.

```python
from tqdm import tqdm

from qatch.models import Tapas

# init the model 
model = Tapas(model_name="google/tapas-large-finetuned-wtq")

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


4. QATCH-Evaluate: Evaluate the results.

```python
from qatch import MetricEvaluator

evaluator = MetricEvaluator(databases=databases)
tests_df = evaluator.evaluate_with_df(tests_df,
                                      prediction_col_name="<prediction_col_name>",
                                      task="QA or SP")
```
The final dataframe contains:

- *db_id*: The database name associated with the test.
- *tbl_name*: The table name associated with the test.
- *sql_tags*: the SQL generator associated with the test.
- *query*: The generated query from step 1.
- *question*: The generated question from step 1. Used as input for the model.
- *predictions_<model_used>*: The predicted query/cells from step 2.
- *5 metrics*: The metrics used to evaluate the models.


# 🏰 Reproduce Experiments

## Install and prepare data

We suggest to create a *data* folder in the project to store all the data but it is not mandatory.
<br> In case the input data are not in this folder, remember to change in *read_data* the *base_path* argument

```bash
mkdir data/
```

These are the tables we use to generate the results in the main paper. <br>
Notice that QATCH perfectly works with any table and the following are only a selected sample to higlight results in the
paper.

 Data               | Link                                                                                                | # rows | # categorical cols | # numerical cols | example cols                 
--------------------|-----------------------------------------------------------------------------------------------------|--------|--------------------|------------------|------------------------------
 Spider             | [link](https://huggingface.co/datasets/spider)                                                      | -      | -                  | -                | -                            
 Sales-transactions | [link](https://www.kaggle.com/datasets/gabrielramos87/an-online-shop-business)                      | 500k   | 5                  | 3                | ProductNo, Date              
 Fitness-trackers   | [link](https://www.kaggle.com/datasets/devsubhash/fitness-trackers-products-ecommerce)              | 565    | 8                  | 3                | Brand Name, Display          
 Account-fraud      | [link](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)            | 1M     | 4                  | 26               | DaysSinceRequest, Velocity6h 
 Late-payment       | [link](https://www.kaggle.com/datasets/hhenry/finance-factoring-ibm-late-payment-histories)         | 2466   | 6                  | 6                | InvoiceDate, Disputed        
 Heart-attack       | [link](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) | 303    | 1                  | 11               | # trtbps, # oldpeak          
 Breast-cancer      | [link](https://www.kaggle.com/datasets/utkarshx27/breast-cancer-dataset-used-royston-and-altman)    | 686    | 5                  | 6                | pgr, rfstime                 
 Adult-census       | [link](https://www.kaggle.com/datasets/uciml/adult-census-income)                                   | 32.6k  | 9                  | 6                | education, fnlwgt            
 Mushrooms          | [link](https://www.kaggle.com/datasets/uciml/mushroom-classification)                               | 8.1k   | 23                 | 0                | cap-shape, ring-type         

# Run Experiments

Current version of QATCH supports SP and QA tasks, however since we rely on third-party models
not all the experiments can be run using QATCH.
Supported models: 

- QA models: Tapas, Tapex, ChatGPT_QA, LLama2_QA and Omnitab
- SP models: ChatGPT_SP and LLama2_SP

For the proprietary data:
```bash
python main_reproducibility.py -gtf proprietary --task QA --model_name Tapas -dsp test_db --inject_null_percentage 0.0
```
For Spider:
```bash
python main_reproducibility.py -gtf spider --task QA --model_name Tapas -dsp test_db --inject_null_percentage 0.0
```
Instead, for the not supported models (because an API does not exist),
the only difference is that the prediction phase has to be done by the user.



