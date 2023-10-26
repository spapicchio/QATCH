
<p align="center">
 <kbd>
  <img src="https://github.com/spapicchio/repository-images/blob/master/QATCH/Verticale.jpg" alt="QATCH's logo" height="300" style="border-radius:50%">
 </kbd>
 </p>

# Q<span style="font-size:0.8em;">A</span>TC<span style="font-size:0.8em;">H</span>: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data

This repository is the official implementation of [QATCH: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data](). 

# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview
* ***What is QATCH?*** **Q**uery-**A**ided **T**RL **Ch**ecklist (QATCH) is a toolbox to highlight TRL models’ strengths and weaknesses on prorietary tables.
* ***How does it work?*** Given a pandas dataframe as input, it generates a testing checklist for Question Answering and Semantic Parsing.
* ***More specifically?*** A query generation algorithm crafts tests by means of the expressive power of SQL.
* ***Ok cool, that's it?*** To evaluate the model's predictions, we propose 5 new metrics intra and inter tuple.
* ***Where can I check the results?*** The generated tests along with the predictions and the metric scores can be downloaded [here](https://drive.google.com/uc?export=download&id=1_z8N52QNAHnxpHv54VhbvYu7DbKV6QRv). This is to prevent the costly generation of test results with the openAI API and to build trust in our results.  
 <figure style="text-align:center">
  <img src="https://github.com/spapicchio/repository-images/blob/master/QATCH/fullPipeline.png">
</figure>
<br>
QATCH's automatically generates and evaluates test checklists on TRL models based on the three-step process depicted below and shown in the image above:

1.  *QATCH-Generate*. It generates a set of queries tailored to proprietary data. For each query it formulates both the SQL declaration, its free-text version, and the expected ground truth consisting of table instances. 
The SQL declaration expresses the logical complexity of the query and reflects the presence/absence of specific features peculiar to relational data model such as presence of missing values and duplicate values.

2. *TRL Model Predictions*. It processes the tests for various TRL models and tasks. The current toolbox version supports for Question Answering [TAPAS](https://github.com/google-research/tapas), [TAPEX](https://github.com/microsoft/Table-Pretraining) and [ChatGPT 3.5](https://openai.com/blog/chatgpt) (need the API key).

3. *QATCH-Evaluate*. It evaluates the models outputs according to a set of cross-task performance metrics.

<p align="center">
<img src="https://github.com/spapicchio/repository-images/blob/master/QATCH/measures.png" width="70%">
</p>
The figure above shows examples of the performance metrics on a toy dataset. 
<br>

Given the ground truth result (target) with three tuples over two attributes, we report the metric values for five predictions, coming either from a QA or from the execution of a query in SP. More details can be found in the [metrics](src/metrics) folder

## Project


```shell
|-- metrics
    |-- metric_evaluator.py # wrapper to initialise the user selected metrics
    |-- abstract_metric.py # abstract class to handle common metric methods
    |-- cell_precision_tag.py # implement cell precision tag
    |-- cell_recall_tag.py # implement cell recall tag
    |-- tuple_cardinality_tag.py # implement tuple cardinality tag
    |-- tuple_constraint_tag.py # implement tuple constraint tag
    |-- tuple_order_tag.py # implement tuple order tag
    
|-- models
    |-- abstract_model.py # abstract class to handle common model methods
    |-- tapas.py # implement input processing and the prediction for TAPAS
    |-- tapex.py # implement input processing and the prediction for TAPEX
    |-- chatgpt.py # implement the input processing and the prediction for ChatGPT
   
|-- test_generator
    |-- database_reader
        |-- single_database.py # initialise single database 
        |-- multiple_databases.py # handle multiple single database instances
        |-- spider_reader.py # read an process SPIDER dataset
        |-- utils.py
    |-- sql_generator
        |-- abstract_sql_generator.py # handle common methods for SQL generators
        |-- select_generator.py # implement SELECT tests
        |-- distinct_generator.py # implement DISTINCT tests
        |-- orderby_generator.py # implement ORDERBY tests
        |-- where_generator.py # implement WHERE tests
    |-- abstract_test_generator.py # handle common methods for test generator class
    |-- test_generator.py # generate tests on proprietary tables
    |-- test_generator_spider.py # generate tests on SPIDER tables
```
# ⚡️ Quickstart
## How to use QATCH with my data?
1. Load your input data as pandas df
```python
import pandas as pd
data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)
db_tables = {'olympic-games': table}
```
2. QATCH-Generate: Generates the tests
```python
from src import TestGenerator
# db_tables is a dict where the key is the tbl_name and the value is the pandas dataframe
test_generator = TestGenerator(db_save_path='/content/games',
                               db_name='games',
                               db_tables=db_tables)

tables, tests_df = test_generator.generate()
```
3. TRL Model Predictions: if you want to use any version of Tapas/Tapex for QA in Huggingface or chatGPT you can use the already implemented modules but it is NOT mandatory.
```python
from tqdm import tqdm
from src import Tapex
tapex_model = Tapex(model_name='microsoft/tapex-large-finetuned-wtq')
tqdm.pandas(desc='predictions')
tests_df['predictions'] = tests_df.progress_apply(
    lambda row: tapex_model.predict(
            table=tables[row.tbl_name],
            queries=row.question),
    axis=1
)
```
4. QATCH-Evaluate: Evaluate the results. For SP 
```python
from utils import get_predictions_results_from_dbs
# only for SP ---------
tests_df = get_predictions_results_from_dbs(base_path_db='<output_db_path>',
                                            df=tests_df,
                                            predictions='<prediction_column_name>')
# ---------------------
metric_evaluator = MetricEvaluator(
    metrics=['cell_precision', 'cell_recall', 'tuple_cardinality',
                         'tuple_constraint', 'tuple_order']
    )
tests_df = metric_evaluator.evaluate_with_df(tests_df,
                                             target='query_result',
                                             predictions='predictions')
```

# 🏰 Reproduce Experiments
## Google colab
All the experiments as well as how to use QATCH can be found in the following Google Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SNoy3GZGPWltVS5cL068xAG9YoPS_3_l?usp=sharing)
<br>
if you do not want to use google colab, continue reading.
## Prepare Environment
First step is to set up a python environment. The current version of the code works with python 3.10.x and we officially support 3.10.11.

```bash
conda create -n your_env_name python=3.10.11
conda activate your_env_name
```
Install other requirements:
```bash
pip install -r requirements.txt
```
## Install and prepare data
We suggest to create a *data* folder in the project to store all the data but it is not mandatory.
<br> In case the input data are not in this folder, remember to change in *read_data* the *base_path* argument
```bash
mkdir data/
```
These are the tables we use to generate the results in the main paper. <br>
Notice that QATCH perfectly works with any table and the following are only a selected sample to higlight results in the paper.

Data | Link | # rows | # categorical cols | # numerical cols | example cols
--- | --- | --- | --- | --- | ---
Spider | [link](https://yale-lily.github.io/spider) | - | - | - | -
Sales-transactions | [link](https://www.kaggle.com/datasets/gabrielramos87/an-online-shop-business) | 500k | 5 | 3 | ProductNo, Date
Fitness-trackers | [link](https://www.kaggle.com/datasets/devsubhash/fitness-trackers-products-ecommerce) | 565 | 8 | 3 | Brand Name, Display
Account-fraud | [link](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | 1M | 4 | 26 | DaysSinceRequest, Velocity6h
Late-payment | [link](https://www.kaggle.com/datasets/hhenry/finance-factoring-ibm-late-payment-histories) | 2466 | 6 | 6 | InvoiceDate, Disputed
Heart-attack | [link](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) | 303 | 1 | 11 | # trtbps, # oldpeak
Breast-cancer | [link](https://www.kaggle.com/datasets/utkarshx27/breast-cancer-dataset-used-royston-and-altman) | 686 | 5 | 6 | pgr, rfstime
Adult-census | [link](https://www.kaggle.com/datasets/uciml/adult-census-income)| 32.6k | 9 | 6 | education, fnlwgt
Mushrooms | [link](https://www.kaggle.com/datasets/uciml/mushroom-classification)| 8.1k | 23 | 0 | cap-shape, ring-type 
# Run Experiments
The current version of QATCH supports two tasks, Question Answering (QA) and Semantic Parsing (SP). 
For QA we provide two scripts one for proprietary data and the other for SPIDER. <br> 
For both scripts, the model name is either the tapas/tapex finetuned version presents on huggingface or chatgpt. For the latter, remember to fill in the credentials.json file to call the API.
```bash
python QA_proprietary_data.py --model_name google/tapas-large-finetuned-wtq --db_category ecommerce --verbose
```
```bash
python QA_SPIDER.py --model_name google/tapas-large-finetuned-wtq --spider_base_path data/spider --verbose
```
Instead, for SP we are not able to provide an end to end script because we are using an external model not accessible from an API i.e. [RESDSQL](https://github.com/RUCKBReasoning/RESDSQL). <br>
However, the following scripts contains all the necessary steps to reproduce the results. 
```bash
python SP_RESDSQL.py --verbose
```
# 📝 License
The source code of our project is released, for research purposes only, under the following Common
Law License: [CC BY-NC](LICENSE). Citing directly the Creative Commons License [website](https://creativecommons.org/about/cclicenses/):<br>
*"This license allows reusers to distribute, remix, adapt, and build upon the material 
in any medium or format for noncommercial purposes only, and only so long as attribution is given to the creator."*



