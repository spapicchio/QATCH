from __future__ import annotations

from typing import Any

import pandas as pd

from .abstract_llama2 import AbstractLLama2
from ..utils import _normalize_output_for_QA, linearize_table


class LLama2_QA(AbstractLLama2):
    """
    A Subclass of `AbstractLLama2` which provides functionality to act as a question answering model
    for tabular data.

    Attributes:
        model_name (str): Name of the Llama model.
        hugging_face_token (str, None): Token for the Hugging Face.
        force_cpu (bool, optional): To force usage of cpu. Defaults to False.

    Methods:
        name: Property attribute which returns the model name.
        prompt: Property attribute which provides instructions for the model in a defined format.
        process_input: Converts input data into a format which model can interpret.
        _normalize_output: Normalize the output for question answering.

    Note:
        - The model used in this class is "meta-llama/Llama-2-7b-chat-hf".
        - The prompt contains few-shot examples to improve the QA task results


    Examples:
        >>> import pandas as pd
        >>> from qatch.models import LLama2_QA
        >>>
        >>> data = pd.DataFrame([
        ...     ["John Doe", "123-456-7890"],
        ...     ["Jane Doe", "098-765-4321"]
        ... ], columns=["Name", "Phone Number"])
        >>>
        >>> llama2_qa_instance = LLama2_QA("meta-llama/Llama-2-7b-chat-hf")
        >>> query = "What is John Doe's phone number?"
        >>> answer = llama2_qa_instance.predict(table=data, query=query, tbl_name='Contact Info')
        >>> print(answer)
        [['123-456-7890']]
  """

    @property
    def name(self):
        return 'LLama2_QA'

    @property
    def prompt(self):
        return """\
        <<SYS>> I want you to act as a question answering model for tabular data.
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
        <</SYS>>
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
