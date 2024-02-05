import logging
from collections import defaultdict
from typing import Any

import pandas as pd
from transformers import TapasForQuestionAnswering, TapasTokenizer

from .abstract_model import AbstractModel


class Tapas(AbstractModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str) -> Any | None:
        """
        Processes a given input for the model by first checking if it exceeds the model's maximum token limit.
        If the input is too long, it returns None. Otherwise, it transforms the input into a specific format that the model accepts.

        Note:
            The TAPAS model works specifically with tables that only contain strings and has a model input limit of 512 tokens.

        Args:
          table (pd.DataFrame): The table to analyse in order to retrieve the answer.
          query (str): The query or list of queries to answer.
          tbl_name (str): The name of the table.

        Returns:
            Any | None: The processed input in a tokenized string format ready for model input,
                        or None if the input length exceeds the model limit of 512 tokens.

        Raises:
            ValueError: When the length of the tokenized input exceeds the accepted length of the model.

        Example:
        ```python
        # Assuming `model_process_input` is an instance of a model that implements this method
        data = pd.DataFrame([["John Doe", "123-456-7890"],["Jane Doe", "098-765-4321"]],
                            columns=["Name", "Phone Number"])
        query = "what are all the phone numbers?"
        tbl_name = "Contact Info"
        processed_input = model.process_input(data, query, tbl_name)
        print(processed_input)
        ```
        """
        if table.shape[0] * table.shape[1] > 512:
            return None

        # convert table to string
        table = table.astype(str)
        # tokenize inputs
        try:
            model_input = self.tokenizer(table=table,
                                         queries=query,
                                         padding="max_length",
                                         return_tensors="pt")
        except ValueError as e:
            # we get error when the tokenized input is longer than accepted from model
            logging.warning(e)
            return None

        return model_input.to(self.device)

    def predict_input(self, model_input, table) -> list[Any]:
        outputs = self.model(**model_input)

        model_input.to('cpu')
        outputs = {idx: outputs[idx].cpu().detach() for idx in outputs}
        [model_input[idx].detach() for idx in model_input]

        pred_query_cords, _ = self.tokenizer.convert_logits_to_predictions(
            model_input,
            outputs['logits'],
            outputs['logits_aggregation']
        )
        answers = []
        for tbl_cords in pred_query_cords:
            query_answer = defaultdict(list)
            [query_answer[row].append(table.iat[(row, col)])
             for row, col in tbl_cords]
            answers.extend(list(query_answer.values()))

        del model_input
        del outputs

        return answers
