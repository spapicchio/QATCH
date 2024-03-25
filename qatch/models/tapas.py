from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import pandas as pd
from transformers import TapasForQuestionAnswering, TapasTokenizer

from .abstract_model import AbstractModel


class Tapas(AbstractModel):
    """
    The Tapas class inherits from the AbstractModel and specializes it to parse tables using the TAPAS model.

    Attributes:
        tokenizer (TapasTokenizer): The tokenizer for input preprocessing.
        model (TapasForQuestionAnswering): The model used to answer the queries from the table.

    Note:
        - The model used in this class is `google/tapas-large-finetuned-wtq`.
        - The TAPAS model works specifically with tables that only contain strings
         and has a model input limit of 512 tokens.

    Examples:
        >>>import pandas as pd
        >>>from qatch.models import Tapas
        >>>
        >>> data = pd.DataFrame([
        ...     ["John Doe", "123-456-7890"],
        ...     ["Jane Doe", "098-765-4321"]
        ... ], columns=["Name", "Phone Number"])
        >>>
        >>> tapas_model = Tapas("google/tapas-large-finetuned-wtq")
        >>> query = "What is John Doe's phone number?"
        >>> answer = tapas_model.predict(table=data, query=query, tbl_name='Contact Info')
        >>> print(answer)
        [['123-456-7890']]
    """

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str) -> Any | None:
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
        # Step 1: Pass the model_input to the model for forward propagation to generate outputs.
        outputs = self.model(**model_input)
        # Step 2: Move the model_input tensor to cpu.
        model_input.to('cpu')
        # Step 3: Move the output tensors to CPU and detach them from the computational graph.
        outputs = {idx: outputs[idx].cpu().detach() for idx in outputs}
        # Step 4: Detach any tensors in model_input from the computational graph.
        [model_input[idx].detach() for idx in model_input]

        # Step 5: Retrieve the coordinates from the logits using a tokenizer function.
        pred_query_cords, _ = self.tokenizer.convert_logits_to_predictions(
            model_input,
            outputs['logits'],
            outputs['logits_aggregation']
        )
        # Step 6: Construct a list of answers.
        answers = []
        for tbl_cords in pred_query_cords:
            query_answer = defaultdict(list)
            # For each coordinate set in the predicted query coordinates,
            # construct a dictionary of row-wise query answers.
            [query_answer[row].append(table.iat[(row, col)])
             for row, col in tbl_cords]
            answers.extend(list(query_answer.values()))
        # Step 7: Dispose of potentially memory-heavy variables.
        del model_input
        del outputs
        # Step 8: Return the list of processed query answers derived from the model's predictions.
        return answers
