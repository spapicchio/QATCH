from __future__ import annotations

from typing import Any

import pandas as pd

from .abstract_llama2 import AbstractLLama2


class LLama2_SP_join(AbstractLLama2):
    """
    Implementation of the Llama2 model specialized for semantic parsing (SP)
    with JOIN statements. Inherits from the Abstract Llama2 model class.

    This model processes the provided schemas and queries, and after transformation,
    predicts the appropriate SQL statements.

    Attributes:
        model_name (str): Name of the Llama model.
        hugging_face_token (str, None): Token for the Hugging Face.
        force_cpu (bool, optional): To force usage of cpu. Defaults to False.

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
        >>> llama_sp_join = LLama2_SP_join(model_name="codellama/CodeLlama-7b-Instruct-hf",
        >>>                                hugging_face_token=credentials['hugging_face_token'])
        >>> # you need to specify all the database table schema
        >>> # if you are using QATCH, you can use database.get_all_table_schema_given(db_id='name_of_the_database')
        >>> db_table_schema = {
        ...                    "student": {"name": ["StudentID", "Grade", "PhoneNumbers"]},
        ...                    "customer": {"name": ["CustomerID", "name", "surname"]},
        ...                    "product": {"name": ["ProductID", "CustomerID", "name", "surname", "price"]}
        ...                   }
        >>> query = "which products did Simone buy?"
        >>> llama_sp_join.predict(table=None,
        >>>                       query=query,
        >>>                       tbl_name=["customer", "product"],
        >>>                       db_table_schema=db_table_schema)
        SELECT T1.name, T2.name FROM "customer" as T1 JOIN "product" as T2 WHERE T1.name == "Simone"
    """

    @property
    def name(self):
        return 'LLama2_SP_join_code'

    @property
    def prompt(self):
        return \
            """[INST]I want you to act as a text to SQL model for tabular data.
            I will pass you as prompt: 
            - all the table names and the respective tables schema present in the database 
            - one question. 
            I want you to parse the question into the SQL command.
            The SQL command must be executable over the database.
            Do not write explanations. Do not type commands. 
            REPLY ONLY WITH THE SQL COMMAND.
            This is an Example:
            Database table names: ["body-builder"] 
            Table schema "body-builder": [Name, Surname] 
            Question: "Show all information about each body builder"
            I want you to output:
            "SELECT * FROM "body-builder"" 
            [/INST] 
            [INST] Database table names: ["student"]
            Schema table "body-builder": [StudentID, Grade, PhoneNumbers]
            Question: "what are all the phone numbers?"
            [/INST]
            'SELECT "PhoneNumbers" FROM student'"
            [INST] Database table names: ["student"]
            Schema table "student": [StudentID, Grade, PhoneNumbers]
            Question: "what is the average grade?"
            [/INST]
            "SELECT AVG(Grade) FROM student"
            [INST] Database table names: ["customer", "product"]
            Schema table "customer": [CustomerID, name, surname]
            Schema table "product": [ProductID, CustomerID, name, surname, price]
            Question: "which products did Simone buy?"
            [/INST]
            SELECT T1.name, T2.name FROM "customer" as T1 JOIN "product" as T2 WHERE T1.name == "Simone"
            """

    def process_input(self,
                      table: pd.DataFrame | None,
                      db_table_schema: dict | None,
                      query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        if not db_table_schema:
            raise ValueError('For Semantic Parsing JOIN, it is needed the schema of the database')

        prompts = [f'[INST] Database table names: {list(db_table_schema.keys())}']

        for name, schema in db_table_schema.items():
            prompts.append(f'Schema table "{name}": {schema["name"].tolist()}')

        prompts.append(f'Question: "{query}"\n[/INST]')
        return "\n".join(prompts)

    def _normalize_output(self, text):
        """
        Normalizes the text received from model predictions.

        Simplifies the predicted SQL command by removing any new lines and extra quotations.

        Args:
            text (str): The raw text prediction from the underlying model.

        Returns:
            str: The normalized text.
        """
        return text.replace('\n', '').replace('"', '').strip()
