import logging
from collections import defaultdict, Counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import TapexTokenizer, BartForConditionalGeneration

from abstract_model import AbstractModel


class Tapex(AbstractModel):
    def __init__(self, model_name: str, force_cpu: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu
                                   else "cpu")
        self.tokenizer = TapexTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def _process_input(self, table: pd.DataFrame, queries: list[str] | str) -> Any | None:
        if len(table) * len(table.columns) > 1024:
            return None
        # convert table to string
        table = table.astype(str)
        # process table
        for col in table.columns:
            table[col] = table[col].str.lower()

        # tapex accepts uncased input since it is pre-trained on the uncased corpus
        if isinstance(queries, list):
            queries = [query.lower() for query in queries]
        else:
            queries = queries.lower()
        try:
            model_input = self.tokenizer(table=table, query=queries,
                                         padding=True, return_tensors="pt")
        except ValueError as e:
            # we get error when the tokenized input is longer than accepted from model
            logging.warning(e)
            return None

        if model_input.input_ids.shape[1] > 1024:
            return None

        return model_input.to(self.device)

    def _predict_queries(self, model_input, table) -> list[list[list[str]]]:
        outputs = self.model.generate(**model_input)
        model_input.to('cpu')
        [model_input[idx].detach() for idx in model_input]
        outputs = outputs.detach().cpu()

        # decode back to text
        pred_cells_queries = self.tokenizer.batch_decode(outputs,
                                                         skip_special_tokens=True)
        # the output contains list of string for each query. Manually transform the output
        answers = []
        for pred_query in pred_cells_queries:
            query_ans = self._return_cells_aggr_by_row(table, pred_query)
            answers.append(query_ans)
        return answers

    @staticmethod
    def _return_cells_aggr_by_row(table, pred_query):
        """perform an aggregation by row of the cells"""
        query_ans = defaultdict(list)
        cells: list = pred_query.split(",")
        counted_cells = Counter(cells)
        for cell, count in counted_cells.items():
            row_ids = np.where(table == cell.strip())[0]
            if len(row_ids) == 0:
                # in case the cell is not present in the table
                row_ids = [-1]
            if count > 1:
                # in case the cell is present in multiple rows
                # we return the first "count" rows
                row_ids = row_ids[:count]
            [query_ans[idx].append(cell.strip()) for idx in row_ids]
        return list(query_ans.values())
