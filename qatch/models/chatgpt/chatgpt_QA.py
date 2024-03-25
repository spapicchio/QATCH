from __future__ import annotations

from typing import Any

import pandas as pd
import tiktoken

from .abstract_chatgpt import AbstractChatGPT
from ..utils import _normalize_output_for_QA, linearize_table


class ChatGPT_QA(AbstractChatGPT):
    """
       A Subclass of `AbstractChatGPT` which provides functionality to act as a question answering model
       for tabular data.

       Attributes:
           api_key (str): The API key for the OpenAI client.
           api_org (str, optional): The organization ID for the OpenAI account. Defaults to None.
           model_name (str, optional): The name of the model to use. Defaults to 'gpt-3.5-turbo-0613'.

       Methods:
           name: Property attribute which returns the model name.
           prompt: Property attribute which provides instructions for the model in a defined format.
           process_input: Converts input data into a format which model can interpret.
           _normalize_output: Normalize the output for question answering.

       Note:
           - The model used in this class is "gpt-3.5-turbo-0613" but you can specify any version you want.
           - The prompt contains few-shot examples to improve the QA task results


       Examples:
           >>> import pandas as pd
           >>> from qatch.models import ChatGPT_QA
           >>>
           >>> data = pd.DataFrame([
           ...     ["John Doe", "123-456-7890"],
           ...     ["Jane Doe", "098-765-4321"]
           ... ], columns=["Name", "Phone Number"])
           >>>
           >>> chatgpt_qa_instance =  ChatGPT_QA(api_key=credentials['api_key_chatgpt'],
           >>>                                  api_org=credentials['api_org_chatgpt'],
           >>>                                  model_name="gpt-3.5-turbo-0613")
           >>> query = "What is John Doe's phone number?"
           >>> answer = chatgpt_qa_instance.predict(table=data, query=query, tbl_name='Contact Info')
           >>> print(answer)
           [['123-456-7890']]
     """

    def __init__(self, api_key: str,
                 api_org: str | None,
                 model_name="gpt-3.5-turbo-0613",
                 *args, **kwargs):
        super().__init__(api_key, api_org, model_name,
                         *args, **kwargs)
        self.encoding = tiktoken.encoding_for_model(model_name)

    @property
    def name(self):
        return "ChatGPT_QA"

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

    def process_input(self,
                      table: pd.DataFrame | None,
                      db_table_schema: list | list[list] | None,
                      query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        if table is None:
            raise ValueError('To use ChatGPT for QA, you need to pass the pandas table')
        linearized_table = linearize_table(table)
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
