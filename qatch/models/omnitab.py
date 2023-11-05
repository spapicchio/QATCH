import logging
from typing import Any

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .abstract_model import AbstractModel


class Omnitab(AbstractModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str) -> Any | None:
        if table.shape[0] * table.shape[1] > 1024:
            logging.warning("Input is too long for model")
            return None

        # convert table to string
        table = table.astype(str)
        # process table
        for col in table.columns:
            table[col] = table[col].str.lower()

        query = query.lower()

        try:
            model_input = self.tokenizer(table=table, queries=query, return_tensors="pt")
        except ValueError as e:
            # we get error when the tokenized input is longer than accepted from model
            logging.warning(e)
            return None
        if model_input.input_ids.shape[1] > 1024:
            logging.warning("Input is too long for model")
            return None
        return model_input.to(self.device)

    def predict_input(self, model_input, table) -> list[list[list[str]]]:
        outputs = self.model.generate(**model_input)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
