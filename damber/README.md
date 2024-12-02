# Deprecated
Damber is based on an older version of QATCH.
Soon will be released a more comprehensive version of damber in a different github.
If you still want to reproduce the results, clone the github before the pull with the newer version 1.0

# DAMBER (**D**ata-**AMB**iguity test**ER**)
 <figure style="text-align:center">

</figure>

<p align="center">
 <kbd>
     <img src="../docs_old/img/first_draft_ambiguous_qatch.drawio.png" alt="Damber's pipeline"  style="border-radius:90%">
 </kbd>
 </p>


**DAMBER1 (Data-AMBiguity testER)** is a new pipeline for ambiguous test generation and evaluation tailored to SP on
tabular
data.
DAMBER relies on QATCH to generate and evaluate the tests.

# How to inject Ambiguous questions?

## Generate tests

The first step is to create the tests and inject ambiguous labels

```python

from damber import damber_generate_ambiguous_tests

ambiguous_df = damber_generate_ambiguous_tests(db_save_path='<base_path_to_store_db_and_tests>',
                                               input_path_tables='damber/data/ambiguous_tables.xlsx',
                                               ambiguity_annotation_file='damber/data/ambiguity_annotation_task_1.xlsx',
                                               ambiguity_th=2.0)

# the file is saved in f'{db_save_path}/ambiguous_tests_df.json'
```

The final dataframe contains:

- *db_id*: The database name associated with the test.
- *tbl_name*: The table name associated with the test.
- *sql_tags*: the SQL generator associated with the test.
- *query*: The generated query from step 1.
- *question*: The generated question from step 1. Used as input for the model.
- *ambiguous_questions*: The ambiguos question generated from question swapping the "ambiguous_labels" with the "
  swapped_attributes"
- *swapped_attribute*: The attribute in "question" to swap
- *ambiguous_labels*: The swapped label
- *ambiguous_attributes*: All the table attributes may be confused with the ambiguous label
- *target_queries*: All the queries that are possible answer for the ambiguous question
  Example:

```json
  {
  "db_id": "WDC_150",
  "tbl_name": "WDC_150",
  "sql_tags": "ORDERBY-SINGLE",
  "query": "SELECT * FROM \"WDC_150\" ORDER BY \"$ From Interest Groups That Opposed\" DESC",
  "question": "Show all data ordered by $ From Interest Groups That Supported in descending order for the table WDC_150",
  "ambiguous_questions": "Show all data ordered by amount in descending order for the table WDC_150",
  "swapped_attribute": "$ From Interest Groups That Supported",
  "ambiguous_labels": "amount",
  "ambiguous_attributes": [
    "$ From Interest Groups That Supported",
    "$ From Interest Groups That Opposed"
  ],
  "target_queries": [
    "SELECT * FROM \"WDC_150\" ORDER BY \"$ From Interest Groups That Supported\" DESC",
    "SELECT * FROM \"WDC_150\" ORDER BY \"$ From Interest Groups That Opposed\" DESC"
  ]
}
```

## Evaluate

The last step is to understand how the model is performing on the ambiguous question:

```python
from damber import damber_evaluate

db_save_path = '<base_path_where_db_is_stored>'  # same specified in the previous step
ambiguous_df = damber_evaluate(test_with_predictions_path='file_with_predictions.json',
                               database_path=db_save_path,
                               prediction_col_name='the_col_name_containing_the_predictions',
                               save_file_path_json='where_to_save_the_json_file.json')
```

The final dataframe contains the metrics calculated over the target_queries.
More precisely, it is taken the max between the average metrics calculated for each target query.

Example:

```json
  {
  "db_id": "WDC_150",
  "tbl_name": "WDC_150",
  "sql_tags": "ORDERBY-SINGLE",
  "query": "SELECT * FROM \"WDC_150\" ORDER BY \"$ From Interest Groups That Opposed\" DESC",
  "question": "Show all data ordered by $ From Interest Groups That Supported in descending order for the table WDC_150",
  "ambiguous_questions": "Show all data ordered by amount in descending order for the table WDC_150",
  "swapped_attribute": "$ From Interest Groups That Supported",
  "ambiguous_labels": "amount",
  "ambiguous_attributes": [
    "$ From Interest Groups That Supported",
    "$ From Interest Groups That Opposed"
  ],
  "target_queries": [
    "SELECT * FROM \"WDC_150\" ORDER BY \"$ From Interest Groups That Supported\" DESC",
    "SELECT * FROM \"WDC_150\" ORDER BY \"$ From Interest Groups That Opposed\" DESC"
  ],
  "predictions_chatgpt_sp": "SELECT * FROM WDC_150 ORDER BY \"$ From Interest Groups That Supported\" DESC",
  "cell_precision_predictions_chatgpt_sp": 1.0,
  "cell_recall_predictions_chatgpt_sp": 1.0,
  "tuple_cardinality_predictions_chatgpt_sp": 1.0,
  "tuple_constraint_predictions_chatgpt_sp": 1.0,
  "tuple_order_predictions_chatgpt_sp": 1.0
}
```

All the metrics are one because the **predictions_chatgpt_sp** is equal to one of the target_queries

# How have you found ambiguous label?

To inject schema-only data-ambiguity in the generated SP tests, DAMBER relies on a human-curated collection of tables
with ground truth ambiguity information ([PYTHIA](https://github.com/enzoveltri/pythia)).
This corpus consists of a set of relational tables R1, ..., Rk whose schema attributes are annotated with
one or more ambiguous labels.
The preprocessed data can be found in ```damber/data/```:

```shell
damber/data/ambiguity_annotation_task_1.xlsx  # ambiguity based on attribute name
damber/data/ambiguity_annotation_task_3.xlsx  # ambiguity based on attribute name and value
damber/data/ambiguous_tables.xlsx  # ambiguity tables
```

