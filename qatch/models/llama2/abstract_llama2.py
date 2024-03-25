from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from typing import Any

import pandas as pd
import torch
import transformers
from huggingface_hub import login
from transformers import AutoTokenizer

from ..utils import check_prediction_list_dim


# pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

class AbstractLLama2(ABC):
    """This is an abstract class for Llama models.

    Attributes:
        model_name (str): Name of the Llama model.
        hugging_face_token (str, None): Token for the Hugging Face.
        force_cpu (bool): To force usage of cpu.
        tokenizer: Tokenizer from the pretrained model.
        pipeline: Text generation pipeline.
    """

    def __init__(self, model_name: str,
                 hugging_face_token: str | None,
                 force_cpu=False,
                 *args, **kwargs):
        """
        Initialize the Llama model configurations.

        Args:
            model_name (str): Name of the Llama model.
            hugging_face_token (str, None): Token for the Hugging Face.
            force_cpu (bool, optional): To force usage of cpu. Defaults to False.
        """

        super().__init__(*args, **kwargs)
        self.model_name = model_name
        login(token=hugging_face_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, truncation=True, truncation_side='right')
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16 if not force_cpu else torch.float32,
            tokenizer=self.tokenizer,
            device_map={"": 0} if not force_cpu else 'cpu',
            trust_remote_code=True,
            truncation=True
        )

    @property
    @abstractmethod
    def prompt(self):
        """Defines the prompt property in the child classes."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """Defines the name property in the child classes."""
        raise NotImplementedError

    def predict(self,
                table: pd.DataFrame | None,
                query: str,
                tbl_name: str | list[str],
                db_table_schema: dict | None = None) -> list[Any] | list[None]:
        """
        Predict results based on the input data.

        Args:
            table (pd.DataFrame, None): The input table data.
            query (str): The query to base the prediction on.
            tbl_name (str, List[str]): The table name.
            db_table_schema (Dict, None, optional): The table schema. Defaults to None.

        Returns:
            list: The list of results.
        """
        model_input = self.process_input(table, db_table_schema, query, tbl_name)
        # print("\033[32m" + f'{tbl_name}: {len(self.tokenizer.tokenize(self.prompt + model_input))}' + "\033[0m")
        if model_input is None or len(self.tokenizer.tokenize(self.prompt + model_input)) > 2048:
            logging.info(f'Table is too large to be processed {tbl_name}')
            result = None
        else:
            result = self.predict_input(model_input)
            if 'SP' not in self.name:
                # only for QA models
                result = check_prediction_list_dim(result, check_llm=False)
        return result

    def predict_input(self, model_input) -> list[Any]:
        """
        Make a prediction based on the model input.

        Args:
            model_input (any): The input for the model.

        Returns:
            list: The resulting prediction.
        """
        final_prompt = self.prompt + model_input
        sequences = self.pipeline(
            final_prompt,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=2048,
            batch_size=1,
        )
        text = sequences[0]['generated_text']
        text = text.replace(final_prompt, '').strip()
        return self._normalize_output(text)

    @abstractmethod
    def _normalize_output(self, text):
        """Provides the way to normalize output in the child classes."""
        raise NotImplementedError

    @abstractmethod
    def process_input(self,
                      table: pd.DataFrame | None,
                      db_table_schema: dict | None,
                      query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        """
        Defines the way to process input in the child classes.

        Args:
            table (pd.DataFrame, None): The input table data.
            db_table_schema (Dict, None, optional): The table schema. Defaults to None.
            query (str): The query to base the input processing on.
            query_tbl_name (str, List[str]): The query table name.

        Returns:
            Any: The processed data.
        """
        raise NotImplementedError
