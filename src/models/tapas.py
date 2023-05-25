import logging
from collections import defaultdict
from typing import Any

import pandas as pd
import torch
from transformers import TapasForQuestionAnswering, TapasTokenizer

from models.abstract_model import AbstractModel


class Tapas(AbstractModel):
    def __init__(self, model_name: str,
                 force_cpu: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu
                                   else "cpu")
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

    def _process_input(self, table: pd.DataFrame, queries: list[str] | str) -> Any | None:
        """
        Processes the input to the model.
        - TAPAS works with a table containing only string.
        - The model input cannot be longer than 512 tokens
        return NONE if the combination of queries and table is too long to be tokenized
        :param table: table to analize to retrieve the answer
        :param queries: list or single query to answer
        :return: processed input or NONE if the input is too long
        """
        if table.shape[0] * table.shape[1] > 512:
            return None

        # convert table to string
        table = table.astype(str)
        # tokenize inputs
        try:
            model_input = self.tokenizer(table=table,
                                         queries=queries,
                                         padding="max_length",
                                         return_tensors="pt")
        except ValueError as e:
            # we get error when the tokenized input is longer than accepted from model
            logging.warning(e)
            return None

        if len(model_input.input_ids[0]) > 512:
            return None

        return model_input.to(self.device)

    def _predict_queries(self, model_input: Any,
                         table: pd.DataFrame) -> list[list[list[str]]]:
        outputs = self.model(**model_input)

        pred_query_cords, _ = self.tokenizer.convert_logits_to_predictions(
            model_input,
            outputs['logits'],
            outputs['logits_aggregation']
        )
        answers = []
        for tbl_cords in pred_query_cords:
            query_answer = defaultdict(list)
            [query_answer[row].append(table.iat[(row, col)])
             for row, col in tbl_cords]
            answers.append(list(query_answer.values()))
        return answers
