# Q<span style="font-size:0.8em;">A</span>TC<span style="font-size:0.8em;">H</span>: Benchmarking Table Representation Learning Models on Your Data

This repository is the official implementation of [QATCH: Benchmarking Table Representation Learning Models on Your Data](). 

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

# ⚡️ Quickstart

## Prepare Environment
