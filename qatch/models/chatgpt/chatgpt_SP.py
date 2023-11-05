from typing import Any

import pandas as pd

from .abstract_chatgpt import AbstractChatGPT


class ChatGPT_SP(AbstractChatGPT):

    def __init__(self, api_key: str,
                 api_org: str | None,
                 model_name="gpt-3.5-turbo-0613",
                 *args, **kwargs):
        super().__init__(api_key, api_org, model_name,
                         *args, **kwargs)

    @property
    def prompt(self):
        return [
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

    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str,
                      ) -> Any | None:
        if tbl_name is None:
            raise ValueError('For Semantic Parsing, it is need the table name '
                             'for the chatgpt input prompt')
        schema = table.columns.tolist()
        prompt = f'Table Name: "{tbl_name}",\nSchema: {schema},\nQuestion: "{query}"'
        return {"role": "user", "content": prompt}

    def _normalize_api_output(self, api_output):
        prediction: str = api_output.choices[0].message.content
        return prediction
