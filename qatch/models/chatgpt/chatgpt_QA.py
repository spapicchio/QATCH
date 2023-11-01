import ast
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import tiktoken

from .abstract_chatgpt import AbstractChatGPT


class ChatGPT_QA(AbstractChatGPT):

    def __init__(self, api_key: str,
                 api_org: str | None,
                 model_name="gpt-3.5-turbo-0613",
                 *args, **kwargs):
        super().__init__(api_key, api_org, model_name,
                         *args, **kwargs)
        self.encoding = tiktoken.encoding_for_model(model_name)

    @property
    def prompt(self):
        return [
            {"role": "user", "content":
                """I want you to act as a question answering model for tabular data.
                   I will pass you a table with one question.
                   I want you to only reply with the output of the question executed on the table.
                   I want you to return the answer in format: list of list (row and columns).
                   The answer must be complete of all the data from the table.
                   If an aggregations is present, return only the aggregate values.
                   Do not write explanations. Do not type commands.
                   This is an Example:
                   Table:
                    [
                        [['Simone', '[H] Name'], ['Papicchio', '[H] Surname']],
                         [['Marco', '[H] Name'], ['Clemente', '[H] Surname']]
                    ],
                    Question: 'Show all information about each body builder']
                    I want you to output:
                    [['Simone', 'Papicchio'], ['Marco', 'Clemente']]
                    """},
            {"role": "user", "content":
                "Table:[[['24172', '[H] Student ID'], ['30', '[H] Grade'], ['3431223445', '[H] Phone Numbers']], "
                "[['281811', '[H] Student ID'], ['22', '[H] Grade'], ['3435227445', '[H] Phone Numbers']]] "
                "Question: 'what are all the phone numbers?'"},
            {"role": "assistant",
             "content": "[['3431223445'], ['3435227445']]"},
            {"role": "user", "content":
                "Table:[[['24172', '[H] Student ID'], ['28', '[H] Grade'], ['3431223445', '[H] Phone Numbers']], "
                "[['281811', '[H] Student ID'], ['24', '[H] Grade'], ['3435227445', '[H] Phone Numbers']]] "
                "Question: 'what is the average of the grade?'"},
            {"role": "assistant",
             "content": "[[26]]"}
        ]

    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str,
                      ) -> Any | None:
        linearized_table = self.linearize_table(table)
        prompt = f"Table: {linearized_table},\nQuestion: '{query}'"
        num_tokens = self._num_tokens_from_string(prompt)
        if num_tokens > 4098:
            self.logger.error('prompt cannot be passed num_tokens > 4098')
            return None
        else:
            return {"role": "user", "content": prompt}

    def _num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def _normalize_api_output(self, api_output):
        prediction: str = api_output.choices[0].message.content
        prediction: list = self._normalize_output_for_QA(prediction)
        return prediction

    @staticmethod
    def _normalize_output_for_QA(prediction: str) -> list[list[Any]] | None:
        # if prediction is None:
        #     return None
        # if prediction == [None]:
        #     return None
        # if prediction == '[]':
        #     return []
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
            return None
        except SyntaxError:
            return None
        except TypeError:
            return None
        prediction = check_prediction_list_dim(prediction)
        return prediction


def check_prediction_list_dim(prediction: list):
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

    prediction = [[cell for cell in row if '[H]' not in str(cell)] for row in prediction]
    prediction = [row for row in prediction if len(row) > 0]
    prediction_len = [len(row) for row in prediction]
    # keep only the row with equal len after removing ['H']
    prediction = [row for row in prediction if len(row) == min(prediction_len)]
    return prediction
