import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any

import openai.error
import pandas as pd
from tqdm import tqdm


class AbstractModel(ABC):
    def predict(self, table: pd.DataFrame,
                queries: list[str] | str,
                max_query_processing_num: int = 5,
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
                model_input = self._process_input(table, _slice, tbl_name)
                if model_input is None:
                    result.append([None] * len(_slice))
                else:
                    try:
                        predictions = self._predict_queries(model_input, table)
                    except openai.error.RateLimitError as e:
                        logging.error(e)
                        return list(itertools.chain(*result))
                    except openai.error.APIError as e:
                        logging.error(e)
                        return list(itertools.chain(*result))
                    except openai.error.InvalidRequestError as e:
                        logging.error(e)
                        if len(_slice) == 1:
                            return [None]
                        else:
                            return result.append([None] * len(_slice))

                    assert len(predictions) == len(_slice)
                    result.append(predictions)
            # Flatten result list
            result = list(itertools.chain(*result))
        else:
            model_input = self._process_input(table, queries, tbl_name)
            if model_input is None:
                return [None] * len(queries)
            try:
                result = self._predict_queries(model_input, table)
            except openai.error.RateLimitError as e:
                logging.error(e)
                return []
            except openai.error.APIError as e:
                logging.error(e)
                return []
            except openai.error.InvalidRequestError as e:
                logging.error(e)
                return []
        return result

    @abstractmethod
    def _process_input(self, table: pd.DataFrame,
                       queries: list[str] | str,
                       tbl_name: str | None = None) -> Any | None:
        """
        process the input before passing it as input to the model
        :param table: table to analize to retrieve the answer
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
