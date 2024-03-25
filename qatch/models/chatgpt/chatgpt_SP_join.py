from __future__ import annotations

from typing import Any

import pandas as pd

from .abstract_chatgpt import AbstractChatGPT


class ChatGPT_SP_join(AbstractChatGPT):
    """
    Implementation of the Llama2 model specialized for semantic parsing (SP)
    with JOIN statements. Inherits from the Abstract Llama2 model class.

    This model processes the provided schemas and queries, and after transformation,
    predicts the appropriate SQL statements.

    Attributes:
        api_key (str): The API key for the OpenAI client.
        api_org (str, optional): The organization ID for the OpenAI account. Defaults to None.
        model_name (str, optional): The name of the model to use. Defaults to 'gpt-3.5-turbo-0613'.

    Methods:
        name: Property attribute which returns the model name.
        prompt: Property attribute which provides instructions for the model in a defined format:
                Database table names: ["customer", "product"], Schema table "customer": [CustomerID, name, surname]
                Schema table "product": [ProductID, CustomerID, name, surname, price]
                Question: "which products did Simone buy?"
        process_input: Processes given inputs into a form that model can consume.
                       Extracts and structures relevant data for the SP task.
        _normalize_output: Normalizes the text received from model predictions.
                           Strips away unnecessary characters from the result SQL statement.

    Note: For this model, the `table` parameter in predict and process_input methods
            is not used and can be set to None.

    Examples:
        >>> chatgpt_sp_join = ChatGPT_SP_join(api_key=credentials['api_key_chatgpt'],
           >>>                                  api_org=credentials['api_org_chatgpt'],
           >>>                                  model_name="gpt-3.5-turbo-0613")
        >>> # you need to specify all the database table schema
        >>> # if you are using QATCH, you can use database.get_all_table_schema_given(db_id='name_of_the_database')
        >>> db_table_schema = {
        ...                    "student": {"name": ["StudentID", "Grade", "PhoneNumbers"]},
        ...                    "customer": {"name": ["CustomerID", "name", "surname"]},
        ...                    "product": {"name": ["ProductID", "CustomerID", "name", "surname", "price"]}
        ...                   }
        >>> query = "which products did Simone buy?"
        >>> chatgpt_sp_join.predict(table=None,
        >>>                       query=query,
        >>>                       tbl_name=["customer", "product"],
        >>>                       db_table_schema=db_table_schema)
        SELECT T1.name, T2.name FROM "customer" as T1 JOIN "product" as T2 WHERE T1.name == "Simone"

    """

    def __init__(self, api_key: str,
                 api_org: str | None,
                 model_name="gpt-3.5-turbo-0613",
                 *args, **kwargs):
        super().__init__(api_key, api_org, model_name,
                         *args, **kwargs)

    @property
    def name(self):
        return 'ChatGPT_SP_join'

    @property
    def prompt(self):
        return [
            {"role": "user", "content":
                """I want you to act as a text to SQL model for tabular data.
                   I will pass you as prompt: 
                   - all the table names and the respective tables schema present in the database 
                   - one question. 
                   I want you to parse the question into the SQL command.
                   The SQL command must be executable over the database.
                   Do not write explanations. Do not type commands. 
                   REPLY ONLY WITH THE SQL COMMAND.
                   This is an Example:
                    Database table names: ["body-builder"], 
                    Table schema "body-builder": [Name, Surname], 
                    Question: "Show all information about each body builder"
                    I want you to output:
                    "SELECT * FROM "body-builder""
                    """},

            {"role": "user", "content":
                'Database table names: ["student"],'
                'Schema table "body-builder": [StudentID, Grade, PhoneNumbers]'
                'Question: "what are all the phone numbers?"'},
            {"role": "assistant",
             "content": 'SELECT "PhoneNumbers" FROM student'},

            {"role": "user", "content":
                'Database table names: ["student"],'
                'Schema table "student": [StudentID, Grade, PhoneNumbers]'
                'Question: "what is the average grade?"'},
            {"role": "assistant",
             "content": "SELECT AVG(Grade) FROM student"},

            {"role": "user", "content":
                'Database table names: ["customer", "product"]'
                'Schema table "customer": [CustomerID, name, surname]'
                'Schema table "product": [ProductID, CustomerID, name, surname, price]'
                'Question: "which products did Simone buy?"'},
            {"role": "assistant",
             "content": 'SELECT T1.name, T2.name FROM "customer" as T1 JOIN "product" as T2 WHERE T1.name == "Simone"'},
        ]

    def process_input(self, table: pd.DataFrame | None,
                      db_table_schema: dict[str, pd.DataFrame], query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        if not db_table_schema:
            raise ValueError('For Semantic Parsing JOIN, it is needed the schema of the database')

        prompts = [f'Database table names: {list(db_table_schema.keys())}']
        for name, schema in db_table_schema.items():
            prompts.append(f'Schema table "{name}": {schema["name"].tolist()}')
        prompts.append(f'Question: "{query}"')

        return {"role": "user", "content": "\n".join(prompts)}

    def _normalize_api_output(self, api_output):
        prediction: str = api_output.choices[0].message.content
        return prediction
