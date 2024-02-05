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
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
        self.name = self.__class__.__name__

    def predict(self,
                table: pd.DataFrame,
                query: str,
                tbl_name: str,
                *args, **kwargs) -> list[Any] | list[None]:
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
