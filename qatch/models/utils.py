from __future__ import annotations

import ast
import logging
import re
from typing import Any

import numpy as np
import pandas as pd


def _normalize_output_for_QA(prediction: str) -> list[list[Any]] | None | str:
    # This initial try/except block is for handling exceptions regarding literal
    # interpretation of the prediction string.
    try:
        # Attempt to evaluate the string of prediction as a Python literal structure (i.e., nested list, tuple, etc).
        prediction = ast.literal_eval(prediction)
    except ValueError as e:
        # Catch ValueError during literal evaluation, log the error message.
        logging.error(e)
    except SyntaxError as e:
        # Catch SyntaxError during literal evaluation, log the error message.
        logging.error(e)
    else:
        # The else block following try/except will only execute if no exceptions were thrown.
        # Apply the helper function check_prediction_list_dim to prediction.
        # check_llm=True implies check for the list length of the prediction.
        prediction = check_prediction_list_dim(prediction, check_llm=True)
        # If no exceptions are thrown, return the prediction result after check_prediction_list_dim.
        return prediction

    # Return None directly if the prediction is None or a list containing None.
    if prediction is None or prediction == [None]:
        return None

    # If the string prediction contains '[H]', extra processes are needed.
    # The '[H]' symbol was likely resulted from the few-shot learning, which is not in the target.
    if '[H]' in prediction:
        # Remove '[H]' and any following word characters with possible white spaces.
        prediction = re.sub(r'\'\[H\](?:\s\w+)?|\',', '', prediction)
        # Replace any numeral followed by comma and single quote with just numeral.
        # This is transforming a single item list to a single item.
        prediction = re.sub(r'(\d+), \'', r'\1', prediction)
        # Replace 'nan,' followed by single quote with 'nan'.
        # This is transforming a single item list to a single item.
        prediction = re.sub(r'(nan), \'', r'\1', prediction)
    # Cache the prediction in case an exception is thrown during the internal parsing below.
    old_prediction = prediction
    try:
        # Remove leading '[' from prediction string until there are none left.
        while len(prediction) > 0 and prediction[0] == '[':
            prediction = prediction[1:]
        # Remove trailing characters of prediction that are not ']'.
        while len(prediction) > 0 and prediction[-1] != ']':
            prediction = prediction[:-1]
        # Remove trailing ']' from prediction string until there are none left.
        while len(prediction) > 0 and prediction[-1] == ']':
            prediction = prediction[:-1]

        # If the prediction string becomes empty after parsing above, return None to represent an empty result.
        if len(prediction) == 0:
            return None

        # Reconstruct prediction as a two-dimensional list-like string.
        prediction = f'[[{prediction}]]'
        # Replace 'nan' with 'None'
        prediction = re.sub(r'nan', r'None', prediction)
        # Literally evaluate the prediction string.
        prediction = eval(prediction)
        # If the result is a tuple, change it to a list.
        # Because we prefer using list for its mutability in Python.
        if isinstance(prediction, tuple):
            new_pred = []
            [new_pred.extend(p) for p in prediction]
            prediction = new_pred

    # If anything wrong occurs during the internal parsing above, directly return the cached old_prediction.
    except NameError:
        return old_prediction
    except SyntaxError:
        return old_prediction
    except TypeError:
        return old_prediction
    # Apply the helper function check_prediction_list_dim to prediction for potentially the last time before output.
    prediction = check_prediction_list_dim(prediction, check_llm=True)
    # Return the final formatted prediction.
    return prediction


def check_prediction_list_dim(prediction: list, check_llm: bool = False
                              ) -> list[list[Any]] | None:
    try:
        # Tries to convert 'prediction' into a numpy array,
        # it may fail if sub-lists in 'prediction' are not of the same length.
        prediction = np.array(prediction)
    except ValueError:
        # If the conversion fails, return None
        return None

    if len(prediction.shape) > 2:
        # If the 'prediction' array is more than 2-dimensional,
        # it needs to be converted into a 2D array.

        if 1 not in prediction.shape:
            # If the shape of the 'prediction' does not
            # contain 1, reshape it into a 2D array.
            rows = prediction.shape[0]
            prediction = prediction.reshape(rows, -1)

        while 1 in prediction.shape:
            # If the shape of 'prediction' contains 1,
            # use np.squeeze to reduce its dimensionality. It is done until
            # there is no dimension with a size of 1 in 'prediction'.
            axes = [ax for ax, x in enumerate(prediction.shape) if x == 1]
            prediction = np.squeeze(prediction, axis=axes[0])

        if prediction.shape == ():
            # If 'prediction' is a scalar (0-dimensional), convert it into a list of list.
            prediction = [[prediction.tolist()]]
        elif len(prediction.shape) == 1:
            # If 'prediction' is 1-dimensional, each element of 'prediction' is put into its own list.
            prediction = [[x] for x in prediction]
        else:
            # If 'prediction' is 2-dimensional, convert it into a list of lists.
            prediction = prediction.tolist()
    elif len(prediction.shape) == 1:
        # If 'prediction' is 1-dimensional, each element of 'prediction' is put into its own list.
        prediction = [[x] for x in prediction]
    elif len(prediction.shape) == 0:
        # If 'prediction' is a scalar (0-dimensional), convert it into a list of list.
        prediction = [[prediction.tolist()]]
    else:
        # If 'prediction' is 2-dimensional, convert it into a list of lists.
        prediction = prediction.tolist()

    if check_llm:
        # Removes rows in 'prediction' that contain the string '[H]' and
        # then keeps only the rows with equal length after removing '[H]'
        prediction = [[cell for cell in row if '[H]' not in str(cell)] for row in prediction]
        prediction = [row for row in prediction if len(row) > 0]
        prediction_len = [len(row) for row in prediction]
        prediction = [row for row in prediction if len(row) == min(prediction_len)]
    return prediction


def linearize_table(table: pd.DataFrame) -> list[list[list[str]]]:
    """
    Linearize a table into a string
        * create a list for each row
        * create a list for each cell passing the content of the cell
          and the header of the cell (with [H])
    """
    columns = table.columns.tolist()
    linearized_table = [
        [
            [row[col], f"[H] {col}"]
            for col in columns
        ]
        for _, row in table.iterrows()
    ]
    return linearized_table
