import logging
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .abstract_model import AbstractModel


class Omnitab(AbstractModel):
    def __init__(self, model_path: str, force_cpu: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu
                                   else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)

    def _process_input(self, table: pd.DataFrame, queries: list[str] | str, tbl_name: str | None = None) -> Any | None:
        if table.shape[0] * table.shape[1] > 1024:
            logging.warning("Input is too long for model")
            return None

        # convert table to string
        table = table.astype(str)
        # process table
        for col in table.columns:
            table[col] = table[col].str.lower()

        if isinstance(queries, list):
            queries = [query.lower() for query in queries]
        else:
            queries = queries.lower()

        try:
            model_input = self.tokenizer(table=table, queries=queries, return_tensors="pt")
        except ValueError as e:
            # we get error when the tokenized input is longer than accepted from model
            logging.warning(e)
            return None
        if model_input.input_ids.shape[1] > 1024:
            logging.warning("Input is too long for model")
            return None

        return model_input.to(self.device)

    def _predict_queries(self, model_input, table) -> list[list[list[str]]]:
        outputs = self.model.generate(**model_input)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
