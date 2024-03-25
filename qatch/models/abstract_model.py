from __future__ import annotations

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
        Runs prediction on the model.

        Args:
            table (pd.DataFrame): The data in table format.
            query (str): The query.
            tbl_name (str): The table name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[Any] | list[None]: List containing the predictions, None if the table is too large to process.

        Raises:
            NotImplementedError: If one of the abstract methods `process_input` or `predict_input` is not properly implemented in the child class.
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
        """
        Processes the input to prepare it for prediction.
        This method must be implemented in the child class.

        Args:
            table (pd.DataFrame): The data in table format.
            query (str): The query.
            tbl_name (str): The table name.

        Returns:
            Any | None: Processed input, or None if the table is too large to process.

        Raises:
            NotImplementedError: If the method is not implemented in the child class.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_input(self, model_input, table) -> list[Any]:
        """
        Runs prediction on the processed input.
        This method must be implemented in the child class.

        Args:
            model_input (Any): The processed model input.
            table (pd.DataFrame): The data in table format.

        Returns:
            list[Any]: List containing the predictions.

        Raises:
            NotImplementedError: If the method is not implemented in the child class.
        """
        raise NotImplementedError
