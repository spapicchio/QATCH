from typing import Any

import pandas as pd

from .abstract_llama2 import AbstractLLama2
from ..utils import _normalize_output_for_QA


class LLama2_QA(AbstractLLama2):
    @property
    def prompt(self):
        return """\
        [INST] I want you to act as a question answering model for tabular data.
        I will pass you a table with one question. 
        I want you to only reply with the output of the question executed on the table.
        I want you to return the answer in format: list of list (row and columns).
        The answer must be complete of all the data from the table.
        If an aggregations is present, return only the aggregate values.
        Do not write explanations. Do not type commands. 
        [/INST]
        ok pass me the input
        [INST] Table Name: "Body_Builders"
        Table: "[[['Simone', '[H] Name'], ['Papicchio', '[H] Surname']], [['Marco', '[H] Name'], ['Clemente', '[H] Surname']]]"
        Question: "Show all information about each body builder"
        [/INST]
        [[Simone', 'Papicchio'], ['Marco', 'Clemente']]
        [INST]
        Table Name: "Students"
        Table: "[[['24172', '[H] Student ID'], ['30', '[H] Grade'], ['3431223445', '[H] Phone Numbers']], [['281811', '[H] Student ID'], ['22', '[H] Grade'], ['3435227445', '[H] Phone Numbers']]]"
        Question: "what are all the phone numbers?"
        [/INST]
        [['3431223445'], ['3435227445']]
        [INST] Table Name: "Students"
        Table: "[[['24172', '[H] Student ID'], ['28', '[H] Grade'], ['3431223445', '[H] Phone Numbers']], [['281811', '[H] Student ID'], ['24', '[H] Grade'], ['3435227445', '[H] Phone Numbers']]]"
        Question: "what is the average of the grade?"
        [/INST]
        [[26]]
        """

    def process_input(self, table: pd.DataFrame,
                      query: list[str] | str,
                      tbl_name: str) -> Any | None:
        linearized_table = self.linearize_table(table)
        model_input = \
            f"""[INST] Table Name: "{tbl_name}"
            Table: "{linearized_table}"
            Question: "{query}"
            [/INST]"""
        return model_input

    def _normalize_output(self, text):
        return _normalize_output_for_QA(text)
