import ast
import logging
import re
from typing import Any

import numpy as np


def _normalize_output_for_QA(prediction: str) -> list[list[Any]] | None | str:
    try:
        prediction = ast.literal_eval(prediction)
    except ValueError as e:
        # possible error of parsing the output
        logging.error(e)
    except SyntaxError as e:
        logging.error(e)
    else:
        prediction = check_prediction_list_dim(prediction)
        return prediction

    if prediction is None or prediction == [None]:
        return None

    # remove the [H] caused by the few-shot learning
    # if isinstance(prediction, str) and '[H]' in prediction:
    if '[H]' in prediction:
        prediction = re.sub(r'\'\[H\](?:\s\w+)?|\',', '', prediction)
        prediction = re.sub(r'(\d+), \'', r'\1', prediction)
        prediction = re.sub(r'(nan), \'', r'\1', prediction)
    old_prediction = prediction
    try:
        while len(prediction) > 0 and prediction[0] == '[':
            prediction = prediction[1:]
        while len(prediction) > 0 and prediction[-1] != ']':
            prediction = prediction[:-1]
        while len(prediction) > 0 and prediction[-1] == ']':
            prediction = prediction[:-1]
        if len(prediction) == 0:
            return None
        prediction = f'[[{prediction}]]'
        prediction = re.sub(r'nan', r'None', prediction)
        prediction = eval(prediction)
        if isinstance(prediction, tuple):
            new_pred = []
            [new_pred.extend(p) for p in prediction]
            prediction = new_pred
    except NameError:
        return old_prediction
    except SyntaxError:
        return old_prediction
    except TypeError:
        return old_prediction
    prediction = check_prediction_list_dim(prediction, check_llm=True)
    return prediction


def check_prediction_list_dim(prediction: list, check_llm: bool = False
                              ) -> list[list[Any]] | None:
    try:
        # may fail because len of the inside array are not equal
        prediction = np.array(prediction)
    except ValueError:
        return None
    if len(prediction.shape) > 2:
        if 1 not in prediction.shape:
            rows = prediction.shape[0]
            prediction = prediction.reshape(rows, -1)
        while 1 in prediction.shape:
            axes = [ax for ax, x in enumerate(prediction.shape) if x == 1]
            prediction = np.squeeze(prediction, axis=axes[0])
        if prediction.shape == ():
            prediction = [[prediction.tolist()]]
        elif len(prediction.shape) == 1:
            prediction = [[x] for x in prediction]
        else:
            prediction = prediction.tolist()
    elif len(prediction.shape) == 1:
        prediction = [[x] for x in prediction]
    elif len(prediction.shape) == 0:
        prediction = [[prediction.tolist()]]
    else:
        prediction = prediction.tolist()
    if check_llm:
        prediction = [[cell for cell in row if '[H]' not in str(cell)] for row in prediction]
        prediction = [row for row in prediction if len(row) > 0]
        prediction_len = [len(row) for row in prediction]
        # keep only the row with equal len after removing ['H']
        prediction = [row for row in prediction if len(row) == min(prediction_len)]
    return prediction
