import ast
import logging
import os
from typing import Any, Literal

import pandas as pd
import torch
import transformers
from huggingface_hub import login
from transformers import AutoTokenizer

from .abstract_model import AbstractModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

class LLama2(AbstractModel):
    def __init__(self, model_name, hugging_face_token: str, test_type: Literal['QA', 'SP'] = 'QA'):
        self.model_name = model_name
        login(token=hugging_face_token)
        self.test_type = test_type
        self.tokenizer = self.init_tokenizer()
        self.pipeline = self.init_pipeline()
        if test_type == 'QA':
            self.message = \
                """I want you to act as a question answering model for tabular data. 
                I will pass you a table with one question.
                I want you to only reply with the output of the question executed on the table.
                I want you to return the answer in format: list of list (row and columns).
                The answer must be complete of all the data from the table.
                If an aggregations is present, return only the aggregate values.
                Do not write explanations. Do not type commands.
                Now I want to act as a chatbot, I will define the "user" as the client and the
                "assistant" as you chatting with me.
                User:
                "Table Name: Body_Builders"
                "Table: [[['Simone', '[H] Name'], ['Papicchio', '[H] Surname']],
                [['Marco', '[H] Name'], ['Clemente', '[H] Surname']]]
                "Question: 'Show all information about each body builder']"
                Assistant:
                "[[Simone', 'Papicchio'], ['Marco', 'Clemente']]"
    
                User:
                "Table Name: Students"
                "Table: [[['24172', '[H] Student ID'], ['30', '[H] Grade'], ['3431223445', '[H] Phone Numbers']], [['281811', '[H] Student ID'], ['22', '[H] Grade'], ['3435227445', '[H] Phone Numbers']]]
                Question: 'what are all the phone numbers?'"
                Assistant:
                "[['3431223445'], ['3435227445']]"
    
                User:
                "Table Name: Students"
                "Table:[[['24172', '[H] Student ID'], ['28', '[H] Grade'], ['3431223445', '[H] Phone Numbers']], [['281811', '[H] Student ID'], ['24', '[H] Grade'], ['3435227445', '[H] Phone Numbers']]]
                Question: 'what is the average of the grade?'"
                Assistant:
                "[[26]]"
                
                """
        elif self.test_type == 'SP':
            self.message = \
                """I want you to act as a text to SQL model for tabular data.
                I will pass you the schema of the table and one question.
                I want you to parse the question into the SQL command.
                The SQL command must be executable with the schema of the table.
                Do not write explanations. Do not type commands. 
                This is an Example:
                Table name: "body-builder", 
                Schema: "[Name, Surname]", 
                Questions: "Show all information about each body builder"
                Output: SELECT * FROM "body-builder"
                
                User:
                Table Name: "Body_Builders"
                Schema: "[Name, Surname]"
                Question: "Show all information about each body builder"
                Output: SELECT * FROM "Body_Builders"
                
                User:
                Table name: "student"
                Schema: "[StudentID, Grade, PhoneNumbers]"
                Question: "what are all the phone numbers?"
                Output: SELECT "PhoneNumbers" FROM "student"
                
                User:
                Table name: "student"
                Schema: "[StudentID, Grade, PhoneNumbers]"
                Question: "what are all the phone numbers?"
                Output: SELECT AVG("Grade") FROM "student"

                """

    def init_pipeline(self):
        return transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map="auto",
            trust_remote_code=True
        )

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def _process_input(self, table: pd.DataFrame, queries: list[str] | str, tbl_name: str | None = None) -> Any | None:
        prompt = None
        if self.test_type == 'QA':
            linearized_table = self._linearize_table(table)
            prompt = f"""User: "Table Name: {tbl_name}" "Table: {linearized_table} Question: '{queries[0]}' "Assistant:"""
        elif self.test_type == 'SP':
            schema = table.columns.tolist()
            prompt = f"""Table name: "{tbl_name}" Schema: "{schema}" Question: "{queries[0]}" Output:"""
        return prompt

    def _predict_queries(self, model_input, table) -> list[Any]:
        prompt = self.message + model_input
        prompt = prompt.replace('\t', '').replace('\n', '').strip()
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=2048,  # TODO insert 4096
            temperature=0.01,
        )
        text = sequences[0]['generated_text']
        text = text.replace(prompt, '').strip()
        if self.test_type == 'QA':
            try:
                content = ast.literal_eval(text)
            except ValueError as e:
                # possible error of parsing the output
                logging.error(e)
                content = text
            except SyntaxError as e:
                logging.error(e)
                content = text
            return content
        else:
            return text
