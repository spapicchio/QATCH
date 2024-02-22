import numpy as np
import pandas as pd

from database_reader import MultipleDatabases
from qatch import MetricEvaluator


def damber_evaluate(test_with_predictions_path: str,
                    database_path: str,
                    prediction_col_name: str,
                    save_file_path_json: str):
    """Main function to run the model analysis."""
    # load df
    df = load_dataframe(test_with_predictions_path)
    # load db an evaluator
    databases = MultipleDatabases(database_path)
    evaluator = MetricEvaluator(databases=databases)
    # calculates metrics
    df = append_metrics_to_dataframe(df, evaluator, prediction_col_name)
    # save the dataframe
    df.to_json(save_file_path_json, orient='records')


def load_dataframe(path: str) -> pd.DataFrame:
    """load the dataframe from json file"""
    df = pd.read_json(path)
    df.columns = df.columns.str.lower().to_list()
    return df


def append_metrics_to_dataframe(df: pd.DataFrame, evaluator, prediction_col_name: str) -> pd.DataFrame:
    """Calculate and append the metrics to the existing dataframe"""
    df_metrics = (df.apply(
        lambda row: evaluate_best_result(row, evaluator, prediction_col_name),
        axis=1, result_type='expand')
    )
    df = pd.concat([df, df_metrics], axis=1).replace({np.nan: None})
    return df


def evaluate_best_result(row: pd.Series, evaluator, prediction_col_name: str) -> dict:
    """Given the row, analyze the predictions compared for each target query"""
    results = []
    for target_query in row['target_queries']:
        row['query'] = target_query
        result = evaluator.evaluate_single_test_SP(row.to_dict(), prediction_col_name, target_col_name='query')
        results.append(result)
    return max(results, key=calculates_score)

def calculates_score(arr: list) -> float:
    """Calculate the mean of the list. It is used to get the max of the list based on average"""
    arr = [val for val in arr if val is not None]
    average = np.mean(arr)
    return average
