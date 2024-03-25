from __future__ import annotations

from typing import Any

import pandas as pd

from .abstract_chatgpt import AbstractChatGPT


class ChatGPT_SP(AbstractChatGPT):
    """
    A Subclass of `AbstractChatGPT` which provides functionality to act as a semantic parsing model for tabular data.

    Attributes:
        api_key (str): The API key for the OpenAI client.
        api_org (str, optional): The organization ID for the OpenAI account. Defaults to None.
        model_name (str, optional): The name of the model to use. Defaults to 'gpt-3.5-turbo-0613'.

    Methods:
        name: Property attribute which returns the model name.
        prompt: Property attribute which provides instructions for the model in a defined format: Table name: "body-builder",
            Schema: "[Name, Surname]", Questions: "Show all information about each body builder"
        process_input: Converts input data into a format which model can interpret.
        _normalize_output: Normalize the output for question answering.

    Note:
       - The model used in this class is "gpt-3.5-turbo-0613" but you can specify any version you want.
       - The prompt contains few-shot examples to improve the QA task results


    Examples:
        >>> import pandas as pd
        >>> from qatch.models import ChatGPT_SP
        >>>
        >>> data = pd.DataFrame([
        ...     ["John Doe", "123-456-7890"],
        ...     ["Jane Doe", "098-765-4321"]
        ... ], columns=["Name", "Phone Number"])
        >>>
        >>> chatgpt_sp_instance = ChatGPT_SP(api_key=credentials['api_key_chatgpt'],
           >>>                                  api_org=credentials['api_org_chatgpt'],
           >>>                                  model_name="gpt-3.5-turbo-0613")
        >>> query = "What is John Doe's phone number?"
        >>> answer = chatgpt_sp_instance.predict(table=data, query=query, tbl_name='Contact Info')
        >>> print(answer)
        SELECT "Phone Number" FROM "Contact Info" WHERE "Name" = "John Doe"
    """

    def __init__(self, api_key: str,
                 api_org: str | None,
                 model_name="gpt-3.5-turbo-0613",
                 *args, **kwargs):
        super().__init__(api_key, api_org, model_name,
                         *args, **kwargs)

    @property
    def name(self):
        return 'ChatGPT_SP'

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

    def process_input(self,
                      table: pd.DataFrame | None,
                      db_table_schema: list | list[list] | None,
                      query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        if not query_tbl_name:
            raise ValueError('For Semantic Parsing, it is need the table name '
                             'for the chatgpt input prompt')

        schema = table.columns.tolist()
        prompt = f'Table Name: "{query_tbl_name}",\nSchema: {schema},\nQuestion: "{query}"'
        return {"role": "user", "content": prompt}

    def _normalize_api_output(self, api_output):
        prediction: str = api_output.choices[0].message.content
        return prediction
