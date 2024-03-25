from __future__ import annotations

from typing import Any

import pandas as pd

from .abstract_llama2 import AbstractLLama2


class LLama2_SP(AbstractLLama2):
    """
    A Subclass of `AbstractLLama2` which provides functionality to act as a semantic parsing model for tabular data.

    Attributes:
        model_name (str): Name of the Llama model.
        hugging_face_token (str, None): Token for the Hugging Face.
        force_cpu (bool, optional): To force usage of cpu. Defaults to False.

    Methods:
        name: Property attribute which returns the model name.
        prompt: Property attribute which provides instructions for the model in a defined format:  Table name: "body-builder",
            Schema: "[Name, Surname]", Questions: "Show all information about each body builder"
        process_input: Converts input data into a format which model can interpret.
        _normalize_output: Normalize the output for question answering.

    Note:
        - The model used in this class is "codellama/CodeLlama-7b-Instruct-hf".
        - The prompt contains few-shot examples to improve the SP task results


    Examples:
        >>> import pandas as pd
        >>> from qatch.models import LLama2_QA
        >>>
        >>> data = pd.DataFrame([
        ...     ["John Doe", "123-456-7890"],
        ...     ["Jane Doe", "098-765-4321"]
        ... ], columns=["Name", "Phone Number"])
        >>>
        >>> llama2_sp_instance = LLama2_SP("codellama/CodeLlama-7b-Instruct-hf")
        >>> query = "What is John Doe's phone number?"
        >>> answer = llama2_sp_instance.predict(table=data, query=query, tbl_name='Contact Info')
        >>> print(answer)
        SELECT "Phone Number" FROM "Contact Info" WHERE "Name" = "John Doe"
    """

    @property
    def name(self):
        return 'LLama2_SP_code'

    @property
    def prompt(self):
        return \
            """[INST] I want you to act as a text to SQL model for tabular data.
            I will pass you the schema of the table and one question.
            I want you to parse the question into the SQL query.
            The SQL command must be executable with the schema of the table.
            Do not write explanations. Do not type commands. 
            [/INST] 
            ok pass me the input
            [INST]
            Table name: "body-builder", 
            Schema: "[Name, Surname]", 
            Questions: "Show all information about each body builder"
            [/INST]
            SELECT * FROM "body-builder"
            [INST] Table Name: "Body_Builders"
            Schema: "[Name, Surname]"
            Question: "Show all information about each body builder"
            [/INST]
            SELECT * FROM "Body_Builders"
            [INST] Table name: "student"
            Schema: "[StudentID, Grade, PhoneNumbers]"
            Question: "what are all the phone numbers?"
            [/INST]
            SELECT "PhoneNumbers" FROM "student"
            [INST] Table name: "student"
            Schema: "[StudentID, Grade, PhoneNumbers]"
            Question: "what are all the phone numbers?"
            [/INST]
            SELECT AVG("Grade") FROM "student"
            """

    def process_input(self,
                      table: pd.DataFrame | None,
                      db_table_schema: dict | None,
                      query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        schema = table.columns.tolist()
        model_input = f"""
        [INST] Table name: "{query_tbl_name}"
        Schema: "{schema}"
        Question: "{query}"
        [/INST] """
        return model_input

    def _normalize_output(self, text):
        return text.replace('\n', '').strip()
