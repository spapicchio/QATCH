import logging
from typing import Any, override

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .abstract_model import AbstractModel


class Omnitab(AbstractModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    @override
    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str) -> Any | None:
        """
        Processes a table and a query into a format that the model can handle.

        The function transforms the table into a string, and tokenizes the table and the query.

        If the input is too long for the model, a warning is logged and the function returns None.

        Args:
            table: The table to process, in pandas DataFrame format.
            query: The query to process, in string format.
            tbl_name: The table name, in string format.

        Returns:
            The processed input, ready to be fed into the model. If the input is too long, the function returns None.
        """
        # Check dimensions of the table (rows*columns). If the size is larger than 1024,
        # we return None with a warning log. This is due to the limitation of the model input size.
        if table.shape[0] * table.shape[1] > 1024:
            logging.warning("Input is too long for model")
            return None

        # The attributes of the DataFrame 'table' are converted to strings to make sure the table
        # representation is in a consistent format.
        table = table.astype(str)

        # The content of the DataFrame 'table' is turned to lowercase in order to standardize the information.
        for col in table.columns:
            table[col] = table[col].str.lower()

        # Also, the query is transformed into lowercase for the same standardization reasons.
        query = query.lower()

        try:
            # The 'table' and 'query' are tokenized using the tokenizer.
            model_input = self.tokenizer(table=table, queries=query, return_tensors="pt")
        except ValueError as e:
            # A ValueError is expected if the tokenized input exceeds the length accepted by the model.
            # If such error is raised, a warning log is returned, and the function returns None.
            logging.warning(e)
            return None

        # Before returning the tokenized input, it checks once again the length of the input.
        # If it's longer than 1024, a warning is logged and the function returns None.
        if model_input.input_ids.shape[1] > 1024:
            logging.warning("Input is too long for model")
            return None

        # If all the above checks pass, the processed input is returned, set to the appropriate device for feeding
        # into the model subsequently.
        return model_input.to(self.device)

    @override
    def predict_input(self, model_input, table) -> list[list[list[str]]]:
        """
        Overrides the parent class method to generate outputs from the model and then decode them back into text.

        Use this method to predict a new output given the processed model inputs.

        Args:
            model_input (TYPE): Input data processed by the process_input method for model to predict.
            table (pd.DataFrame): Original table used in model inputs.

        Returns:
            list[list[list[str]]]: List of decoded model outputs. Each output is represented by a list[list] strings.
        Example:
            >>> model_input = {"input_ids": [101, 1898, 102], "attention_mask": [1, 1, 1]}
            >>> table = pd.DataFrame(...)
            >>> model.predict_input(model_input, table)
            [['hello', 'world'], ['foo', 'bar']]
        """
        outputs = self.model.generate(**model_input)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
