import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import torch

from .utils import check_prediction_list_dim


class AbstractModel(ABC):
    def __init__(self, force_cpu=False, *args, **kwargs):
        # set up logger
        self.logger = logging
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                             and not force_cpu
                                   else "cpu")
        self.name = self.__class__.__name__

    def predict(self,
                table: pd.DataFrame,
                query: str,
                tbl_name: str) -> list[Any] | list[None]:
        """
        """
        model_input = self.process_input(table, query, tbl_name)
        if model_input is None:
            """Table is too large to be processed"""
            result = None
        else:
            result = self.predict_input(model_input, table)
            if 'SP' not in self.name:
                # only for QA models
                result = check_prediction_list_dim(result, check_llm=False)
        return result

    @abstractmethod
    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def predict_input(self, model_input, table) -> list[Any]:
        raise NotImplementedError

    @staticmethod
    def linearize_table(table: pd.DataFrame) -> list[list[list[str]]]:
        """
        Linearize a table into a string
            * create a list for each row
            * create a list for each cell passing the content of the cell
              and the header of the cell (with [H])
        """
        columns = table.columns.tolist()
        linearized_table = [
            [
                [row[col], f"[H] {col}"]
                for col in columns
            ]
            for _, row in table.iterrows()
        ]
        return linearized_table
