from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .abstract_model import AbstractModel


class Omnitab(AbstractModel):
    """
    The Omnitab class inherits from the AbstractModel and specializes it to parse tables using the Omnitab model.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for input preprocessing.
        model (AutoModelForSeq2SeqLM): The model used to answer the queries from the table.

    Note:
        - The model used in this class is 'neulab/omnitab-large-finetuned-wtq'.
        - The Omnitab model works specifically with tables that only contain strings
         and has a model input limit of 1024 tokens.

    Examples:
        >>>import pandas as pd
        >>>from qatch.models import Tapas
        >>>
        >>> data = pd.DataFrame([
        ...     ["John Doe", "123-456-7890"],
        ...     ["Jane Doe", "098-765-4321"]
        ... ], columns=["Name", "Phone Number"])
        >>>
        >>> omnitab_model = Omnitab("'neulab/omnitab-large-finetuned-wtq'")
        >>> query = "What is John Doe's phone number?"
        >>> answer = omnitab_model.predict(table=data, query=query, tbl_name='Contact Info')
        >>> print(answer)
        [['123-456-7890']]
    """

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str) -> Any | None:
        # Check dimensions of the table (rows*columns). If the size is larger than 1024,
        # we return None with a warning log. This is due to the limitation of the model input size.
        if table.shape[0] * table.shape[1] > 1024:
            logging.warning("Input is too long for model")
            return None

        # The attributes of the DataFrame 'table' are converted to strings to make sure the table
        # representation is in a consistent format.
        table = table.astype(str)

        # The content of the DataFrame 'table' is turned to lowercase in order to standardize the information.
        for col in table.columns:
            table[col] = table[col].str.lower()

        # Also, the query is transformed into lowercase for the same standardization reasons.
        query = query.lower()

        try:
            # The 'table' and 'query' are tokenized using the tokenizer.
            model_input = self.tokenizer(table=table, queries=query, return_tensors="pt")
        except ValueError as e:
            # A ValueError is expected if the tokenized input exceeds the length accepted by the model.
            # If such error is raised, a warning log is returned, and the function returns None.
            logging.warning(e)
            return None

        # Before returning the tokenized input, it checks once again the length of the input.
        # If it's longer than 1024, a warning is logged and the function returns None.
        if model_input.input_ids.shape[1] > 1024:
            logging.warning("Input is too long for model")
            return None

        # If all the above checks pass, the processed input is returned, set to the appropriate device for feeding
        # into the model subsequently.
        return model_input.to(self.device)

    def predict_input(self, model_input, table) -> list[list[list[str]]]:
        outputs = self.model.generate(**model_input)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
