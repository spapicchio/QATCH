import ast
import logging
from typing import Any, Literal

import openai
import pandas as pd
import tiktoken

from .abstract_model import AbstractModel


class ChatGPT(AbstractModel):
    def __init__(self, api_key: str,
                 api_org: str | None = None,
                 test_type: Literal['QA', 'SP'] = 'QA'):
        self.encoding = None
        self.api_key = api_key
        openai.organization = api_org
        openai.api_key = api_key
        test_type = test_type.upper()
        self.test_type = test_type
        if test_type == 'QA':
            self.messages = [
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
                        ],"
                        "Questions: 'Show all information about each body builder']
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
                 "content": "[[26]]"},
            ]
        elif test_type == 'SP':
            self.messages = [
                {"role": "user", "content":
                    """I want you to act as a text to SQL model for tabular data.
                       I will pass you the schema of the table and one question.
                       I want you to parse the question into the SQL command.
                       The SQL command must be executable with the schema of the table.
                       Do not write explanations. Do not type commands. 
                       REPLY ONLY WITH THE SQL COMMAND.
                       This is an Example:
                       Table name: "body-builder", 
                        Schema: [Name, Surname], 
                        Questions: "Show all information about each body builder"
                        I want you to output:
                        "SELECT * FROM "body-builder""
                        """},
                {"role": "user", "content":
                    'Table name: "student",'
                    "Schema: [StudentID, Grade, PhoneNumbers]"
                    'Question: "what are all the phone numbers?"'},
                {"role": "assistant",
                 "content": 'SELECT "PhoneNumbers" FROM student'},
                {"role": "user", "content":
                    'Table name: "student"'
                    "Schema: [StudentID, Grade, PhoneNumbers]"
                    'Question: "what is the average grade?"'},
                {"role": "assistant",
                 "content": "SELECT AVG(Grade) FROM student"},
            ]
        else:
            raise ValueError('Task not recognized.')

    def num_tokens_from_string(self, string: str, encoding_name: str = 'gpt-3.5-turbo') -> int:
        """Returns the number of tokens in a text string."""
        if self.encoding is None:
            self.encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def _process_input(self, table: pd.DataFrame, queries: list[str] | str, tbl_name) -> Any | None:
        """
        max_tokens: 4096
        https://platform.openai.com/docs/models/gpt-3-5"""

        if self.test_type == 'QA':
            prompt = self._process_input_QA(table, queries)
        elif self.test_type == 'SP':
            prompt = self._process_input_SP(table, queries, tbl_name)
        else:
            raise ValueError(f"test_type {self.test_type} not supported")

        num_tokens = self.num_tokens_from_string(prompt)
        if num_tokens > 3950:
            logging.error('NUM_TOKENS > 3950')
            return None
        else:
            return {"role": "user", "content": prompt}

    def _process_input_QA(self, table: pd.DataFrame, queries: list[str] | str) -> Any | None:
        linearized_table = self._linearize_table(table)
        return f"Table: {linearized_table}, Question: '{queries[0]}'"

    def _process_input_SP(self, table, queries, tbl_name):
        if tbl_name is None:
            raise ValueError('For Semantic Parsing, it is need the table name '
                             'for the chatgpt input prompt')
        schema = table.columns.tolist()
        return f'Table Name: "{tbl_name}", Schema: {schema}, Question: "{queries[0]}"'

    def _predict_queries(self, model_input, table) -> list[Any]:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=self.messages + [model_input],
                temperature=0.0,  # make it deterministic
                # max_tokens=4097, # max tokens in the generated output
                # top_p=1, ## alternative to temperature
                # frequency_penalty=0,
                # presence_penalty=0
            )
        except openai.error.InvalidRequestError as e:
            # raise error because the input is too long
            logging.error(e)
            return [None]

        if self.test_type == 'QA':
            content = self._predict_QA(response)
        elif self.test_type == 'SP':
            content = self._predict_SP(response)
        return [content]

    def _predict_QA(self, response):
        try:
            content = ast.literal_eval(response.choices[0].message.content)
        except ValueError as e:
            # possible error of parsing the output
            logging.error(e)
            content = response.choices[0].message.content
        except SyntaxError as e:
            logging.error(e)
            content = response.choices[0].message.content
        return content

    def _predict_SP(self, response):
        return response.choices[0].message.content

