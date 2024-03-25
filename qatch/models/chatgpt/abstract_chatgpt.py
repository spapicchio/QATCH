from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import sleep
from typing import Any

import pandas as pd
from openai import OpenAI, BadRequestError, RateLimitError, APIConnectionError


class AbstractChatGPT(ABC):
    def __init__(self, api_key: str, api_org: str | None,
                 model_name="gpt-3.5-turbo-0613",
                 *args, **kwargs):
        # initialize openAI
        self.logger = logging
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, organization=api_org, max_retries=3)
        self.model_name = model_name

    @property
    @abstractmethod
    def prompt(self):
        raise NotImplementedError

    def predict(self, table: pd.DataFrame | None,
                query: str,
                tbl_name: str | list[str],
                db_table_schema: dict | None = None) -> list[Any] | list[None]:
        """wrap function to process input and predict queries
        excepts the openAI errors and return an empty list"""
        # 1. model input
        model_input = self.process_input(table, db_table_schema, query, tbl_name)
        if model_input is None:
            """Table is too large to be processed"""
            return [None]

        pred = ['nan']
        count = 1
        while pred == ['nan']:
            if count > 10:
                self.logger.error('Too many errors, aborting.')
                raise ValueError('Too many errors, aborting.')
            # 2. predict
            pred = self._call_api(model_input)
            if pred == ['nan']:
                # error occurred
                sleep(60)
                self.logger.info('sleep for 60')
                count += 1
        return pred

    def _call_api(self, model_input, table=None) -> list[Any]:
        try:
            # content = self._predict_with_api(model_input)
            content = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.prompt + [model_input],
                temperature=0.0,
                timeout=60 * 3,  # after 3 min (default 10)
            )
            content = self._normalize_api_output(content)
        except BadRequestError as e:
            # raise error because the input is too long
            self.logger.error(e)
            return [None]
        except RateLimitError as e:
            """Too many requests to the API"""
            self.logger.error(e)
            return ['nan']
        except APIConnectionError as e:
            """Too many requests to the API"""
            self.logger.error(e)
            return ['nan']
        else:
            return content

    @abstractmethod
    def _normalize_api_output(self, api_output):
        raise NotImplementedError

    @abstractmethod
    def process_input(self,
                      table: pd.DataFrame | None,
                      db_table_schema: dict | None,
                      query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        raise NotImplementedError
