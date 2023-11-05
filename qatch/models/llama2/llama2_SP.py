from typing import Any

import pandas as pd

from .abstract_llama2 import AbstractLLama2


class LLama2_SP(AbstractLLama2):
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

    def process_input(self, table: pd.DataFrame,
                      query: list[str] | str,
                      tbl_name: str) -> Any | None:
        schema = table.columns.tolist()
        model_input = f"""
        [INST] Table name: "{tbl_name}"
        Schema: "{schema}"
        Question: "{query}"
        [/INST] """
        return model_input

    def _normalize_output(self, text):
        return text.replace('\n', '').strip()
