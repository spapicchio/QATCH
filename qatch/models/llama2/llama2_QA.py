from typing import Any

import pandas as pd

from .abstract_llama2 import AbstractLLama2
from ..utils import _normalize_output_for_QA, linearize_table


class LLama2_QA(AbstractLLama2):
    @property
    def prompt(self):
        return """\
        [INST] I want you to act as a question answering model for tabular data.
        I will pass you a table with one question. 
        I want you to return the elements in the table that answer the question.
        I want you to return the answer in format: list of list (row and columns).
        The answer must be complete of all the data from the table.
        If an aggregations is present, return only the aggregate values.
        The answer must be generated only from the table provided.
        The answer must have the same format of the Table passed as input.
        The answer must be a list of tuples. Then for each tuple, a list of elements. For each element a list of cell values and the header. 
        Do not use different formats in the answer. 
        Do not repeat the instruction in the answer.
        [/INST]
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

    def process_input(self,
                      table: pd.DataFrame | None,
                      db_table_schema: dict | None,
                      query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        if table.size > 512:
            return None
        linearized_table = linearize_table(table)
        model_input = \
            f"""[INST] Table Name: "{query_tbl_name}"
            Table: "{linearized_table}"
            Question: "{query}"
            [/INST]"""
        return model_input

    def _normalize_output(self, text):
        return _normalize_output_for_QA(text)
