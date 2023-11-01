from typing import Any

import pandas as pd
import tiktoken

from .abstract_chatgpt import AbstractChatGPT
from ..utils import _normalize_output_for_QA


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
        prediction: list = _normalize_output_for_QA(prediction)
        return prediction


