import logging
from collections import defaultdict
from typing import Any

import pandas as pd
from transformers import TapasForQuestionAnswering, TapasTokenizer

from .abstract_model import AbstractModel


class Tapas(AbstractModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str) -> Any | None:
        """
        Processes the input to the model.
        - TAPAS works with a table containing only string.
        - The model input cannot be longer than 512 tokens
        return NONE if the combination of queries and table is too long to be tokenized
        :param table: table to analize to retrieve the answer
        :param query: list or single query to answer
        :return: processed input or NONE if the input is too long
        """
        if table.shape[0] * table.shape[1] > 512:
            return None

        # convert table to string
        table = table.astype(str)
        # tokenize inputs
        try:
            model_input = self.tokenizer(table=table,
                                         queries=query,
                                         padding="max_length",
                                         return_tensors="pt")
        except ValueError as e:
            # we get error when the tokenized input is longer than accepted from model
            logging.warning(e)
            return None

        return model_input.to(self.device)

    def predict_input(self, model_input, table) -> list[Any]:
        outputs = self.model(**model_input)

        model_input.to('cpu')
        outputs = {idx: outputs[idx].cpu().detach() for idx in outputs}
        [model_input[idx].detach() for idx in model_input]

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
            answers.extend(list(query_answer.values()))

        del model_input
        del outputs
        
        return answers

