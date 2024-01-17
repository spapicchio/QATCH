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
    def __init__(self, model_name: str,
                 hugging_face_token: str | None,
                 force_cpu=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        login(token=hugging_face_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16 if not force_cpu else torch.float32,
            tokenizer=self.tokenizer,
            device_map={"": 0} if not force_cpu else 'cpu',
            trust_remote_code=True
        )

    @property
    @abstractmethod
    def prompt(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    def predict(self,
                table: pd.DataFrame | None,
                query: str,
                tbl_name: str | list[str],
                db_table_schema: dict | None = None) -> list[Any] | list[None]:
        """"""
        model_input = self.process_input(table, db_table_schema, query, tbl_name)
        if model_input is None:
            """Table is too large to be processed"""
            result = None
        else:
            result = self.predict_input(model_input)
            if 'SP' not in self.name:
                # only for QA models
                result = check_prediction_list_dim(result, check_llm=False)
        return result

    def predict_input(self, model_input) -> list[Any]:
        final_prompt = self.prompt + model_input
        sequences = self.pipeline(
            final_prompt,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=2048,
            batch_size=1
        )
        text = sequences[0]['generated_text']
        text = text.replace(final_prompt, '').strip()
        return self._normalize_output(text)

    @abstractmethod
    def _normalize_output(self, text):
        raise NotImplementedError

    @abstractmethod
    def process_input(self,
                      table: pd.DataFrame | None,
                      db_table_schema: dict | None,
                      query: str,
                      query_tbl_name: str | list[str]) -> Any | None:
        raise NotImplementedError
