import logging
from typing import Any, Literal

import openai
import pandas as pd

from models.abstract_model import AbstractModel


class ConcreteChatGPT(AbstractModel):
    def __init__(self, api_key: str, api_org: str, test_type: Literal['QA', 'SP'] = 'QA'):
        self.api_key = api_key
        openai.organization = api_org
        openai.api_key = api_key
        self.test_type = test_type
        if test_type == 'QA':
            self.messages = [
                {"role": "user", "content":
                    """I want you to act as a question answering model for tabular data.
                       I will pass you a table with one or multiple questions into a list.
                       I want you to only reply with the output of the question run on the table.
                       I want you to return a list containing the answer for each question.
                       Do not write explanations. Do not type commands unless I instruct you to do so.
                       This is an Example:
                       Table:
                        [
                            [['Simone', '[H] Name'], ['Papicchio', '[H] Surname']],
                             [['Marco', '[H] Name'], ['Clemente', '[H] Surname']]
                        ],"
                        "Questions: ['Show all information about each body builder', 'Show information of the body builder Simone']
                        I want you to output:
                        [
                            [['Simone', 'Papicchio'], ['Marco', 'Clemente']],
                            [['Simone', 'Papicchio']]
                        ]
                        """},
                {"role": "user", "content":
                    "Table:[[['24172', '[H] Student ID'], ['30', '[H] Grade'], ['3431223445', '[H] Phone Numbers']], [['281811', '[H] Student ID'], ['22', '[H] Grade'], ['3435227445', '[H] Phone Numbers']]] "
                    "Question: ['What is the average of the grade?', 'what are all the phone numbers?']"},
                {"role": "assistant",
                 "content": "[ [['26']], [['3431223445'], ['3435227445']] ]"},
            ]
        else:
            # TODO: Add SP test
            self.messages = None

    def _process_input(self, table: pd.DataFrame, queries: list[str] | str) -> Any | None:
        if table.shape[0] * table.shape[1] > 300:
            return None
        linearized_table = self._linearize_table(table)
        if isinstance(queries, str):
            queries = [queries]
        prompt = f'Table: {linearized_table}, Questions: {queries}'

        return {"role": "user", "content": prompt}

    def _predict_queries(self, model_input, table) -> list[Any]:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages + [model_input],
            temperature=0.0,  # make it deterministic
            # max_tokens=256, # max tokens in the generated output
            # top_p=1, ## alternative to temperature
            # frequency_penalty=0,
            # presence_penalty=0
        )
        try:
            content = eval(response.choices[0].message.content)
        except SyntaxError as e:
            # TODO: chatgpt now is returning None but we are not considering at all
            # possible error of parsing the output
            logging.warning(e)
            queries = model_input.split('Questions:')[1]
            content = [None] * len(eval(queries))
        return content

    @staticmethod
    def _linearize_table(table: pd.DataFrame) -> list[list[list[str]]]:
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
