from abc import abstractmethod, ABC
from typing import Any

import torch
import transformers
from huggingface_hub import login
from transformers import AutoTokenizer

from ..abstract_model import AbstractModel


# pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

class AbstractLLama2(AbstractModel, ABC):
    def __init__(self, model_name: str,
                 hugging_face_token: str | None = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        login(token=hugging_face_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map={"": 0},
            trust_remote_code=True
        )

    @property
    @abstractmethod
    def prompt(self):
        raise NotImplementedError

    def predict_input(self, model_input, table) -> list[Any]:
        final_prompt = self.prompt + model_input
        sequences = self.pipeline(
            final_prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=2048,  # TODO insert 4096
            temperature=0.01,
        )
        text = sequences[0]['generated_text']
        text = text.replace(final_prompt, '').strip()
        return self._normalize_output(text)

    @abstractmethod
    def _normalize_output(self, text):
        raise NotImplementedError
