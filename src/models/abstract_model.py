import itertools
import logging
from abc import ABC, abstractmethod
from time import sleep
from typing import Any

import openai.error
import pandas as pd
from tqdm import tqdm


class AbstractModel(ABC):
    def predict(self, table: pd.DataFrame,
                queries: list[str] | str,
                max_query_processing_num: int = 1,
                tbl_name=None) -> list[Any] | list[None]:
        """
        predict the answer for multiple queries.
        - if the number of queries is greater than max_query_processing_num,
          the queries are split into slices and predicted separately.
        - return NONE if the combination of queries and table is too long to be tokenized

        :param tbl_name: # TODO update doc
        :param table: table to analize to retrieve the answer
        :param queries: list or single query to answer
        :param max_query_processing_num: maq query to analyze at once,
            ATTENTION the higher is the number the higher is the probability to get NONE as answer
        :return: the answer for each query
        """
        if isinstance(queries, str):
            queries = [queries]

        if len(queries) > max_query_processing_num:
            result = []
            # Split questions into slices of max_query_processing_num
            questions_slices = [queries[i:i + max_query_processing_num]
                                for i in range(0, len(queries), max_query_processing_num)]
            # Predict for each slice
            for _slice in tqdm(questions_slices, desc='predicting'):
                predictions = self._handle_prediction_openai_errors(table,
                                                                    _slice,
                                                                    tbl_name)
                result.append(predictions)
            # Flatten result list
            result = list(itertools.chain(*result))
        else:
            result = self._handle_prediction_openai_errors(table, queries, tbl_name)
        return result[0]

    def _handle_prediction_openai_errors(self, table: pd.DataFrame,
                                         queries: list[str],
                                         tbl_name: str) -> list:
        """handle openAI API errors to avoid loosing predictions until the error"""
        pred = ['nan'] * len(queries)
        start = 0
        count = 1
        while 'nan' in pred:
            if count > 10000:
                logging.error('Too many errors, aborting.')
            new_pred = self.predict_queries(table, queries, tbl_name)
            if len(new_pred) < len(queries):
                pred[start:start + len(new_pred)] = new_pred
                start += len(new_pred)
                sleep(60)
                logging.info('sleep for 60')
                count += 1
            elif len(new_pred) == len(queries):
                pred = new_pred
        return pred

    def predict_queries(self, table: pd.DataFrame,
                        queries: list[str],
                        tbl_name: str) -> list:
        """wrap function to process input and predict queries
        excepts the openAI errors and return an empty list"""
        model_input = self._process_input(table, queries, tbl_name)
        if model_input is None:
            """Table is too large to be processed"""
            return [None] * len(queries)
        try:
            result = self._predict_queries(model_input, table)
        except openai.error.RateLimitError as e:
            """Too many requests to the API"""
            logging.error(e)
            return []
        except openai.error.APIError as e:
            """Too many requests to the API"""
            logging.error(e)
            return []
        except openai.error.ServiceUnavailableError as e:
            """Too many requests to the API"""
            logging.error(e)
            return []
        except openai.error.Timeout as e:
            logging.error(e)
            return []
        return result

    @abstractmethod
    def _process_input(self, table: pd.DataFrame,
                       queries: list[str] | str,
                       tbl_name: str | None = None) -> Any | None:
        """
        process the input before passing it as input to the model
        :param table: table to analyze to retrieve the answer
        :param queries: list or single query to answer
        :return: processed input
        """
        raise NotImplementedError

    @abstractmethod
    def _predict_queries(self, model_input, table) -> list[Any]:
        """
        predict the answer for multiple queries
        :param table:
        :param model_input:
        :return: list of answers
        """
        raise NotImplementedError

    @staticmethod
    def _linearize_table(table: pd.DataFrame) -> list[list[list[str]]]:
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
