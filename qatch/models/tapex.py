from __future__ import annotations

import logging
import warnings
from collections import defaultdict, Counter
from typing import Any

import numpy as np
import pandas as pd
from transformers import TapexTokenizer, BartForConditionalGeneration

from .abstract_model import AbstractModel


class Tapex(AbstractModel):
    """
    The Tapex class inherits from the AbstractModel and specializes it to parse tables using the TAPEX model.

    Attributes:
        tokenizer (TapexTokenizer): The tokenizer for input preprocessing.
        model (BartForConditionalGeneration): The model used to answer the queries from the table.

    Note:
        - The model used in this class is 'microsoft/tapex-large-finetuned-wtq'.
        - The TAPEX model works specifically with tables that only contain strings
         and has a model input limit of 1024 tokens.

    Examples:
        >>> import pandas as pd
        >>> from qatch.models import Tapex
        >>>
        >>> data = pd.DataFrame([
        ...     ["John Doe", "123-456-7890"],
        ...     ["Jane Doe", "098-765-4321"]
        ... ], columns=["Name", "Phone Number"])
        >>>
        >>> tapex_model = Tapex("microsoft/tapex-large-finetuned-wtq")
        >>> query = "What is John Doe's phone number?"
        >>> answer = tapex_model.predict(table=data, query=query, tbl_name='Contact Info')
        >>> print(answer)
        [['123-456-7890']]
    """

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = TapexTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def process_input(self, table: pd.DataFrame,
                      query: str,
                      tbl_name: str) -> Any | None:
        if table.shape[0] * table.shape[1] > 1024:
            return None

        # convert table to string
        table = table.astype(str)
        # process table
        for col in table.columns:
            table[col] = table[col].str.lower()

        # tapex accepts uncased input since it is pre-trained on the uncased corpus
        query = query.lower()
        try:
            model_input = self.tokenizer(table=table, query=query,
                                         padding=True, return_tensors="pt")
        except ValueError as e:
            # we get error when the tokenized input is longer than accepted from model
            logging.warning(e)
            return None

        if model_input.input_ids.shape[1] > 1024:
            warnings.warn(f'After tokenization'
                          f' the input is longer than 1024 tokens: '
                          f'{model_input.input_ids.shape[1]}. '
                          'the input will be skipped')
            return None

        return model_input.to(self.device)

    def predict_input(self, model_input, table) -> list[Any]:
        outputs = self.model.generate(**model_input)
        model_input.to('cpu')

        [model_input[idx].detach() for idx in model_input]
        outputs = outputs.detach().cpu()

        # decode back to text
        pred_cells_queries = self.tokenizer.batch_decode(outputs,
                                                         skip_special_tokens=True)
        # the output contains list of string for each query. Manually transform the output
        answers = []
        for pred_query in pred_cells_queries:
            query_ans = self._return_cells_aggr_by_row(table, pred_query)
            answers.extend(query_ans)
        del model_input
        del outputs
        return answers

    @staticmethod
    def _return_cells_aggr_by_row(table, pred_query):
        """
        Perform an aggregation operation by row of the cells in a specified table based on a predicate query.

        Args:
            table (np.array): The table to perform the operation on.
            pred_query (str): The predicate query used for aggregation operation. It should be a string of comma
                              separated cell values, e.g., "cell1,cell2,cell3".

        Returns:
            list: Returns a list of lists where each sublist contains aggregated cell values from a single row of the table.

        Example:
            Let's assume we have a table as below:

            [["cell1", "cell2"],
            ["cell3", "cell1"],
            ["cell1", "cell2"]]

            And pred_query as "cell1,cell1,cell2"

            Calling _return_cells_aggr_by_row(table, pred_query) will give:

            [["cell1", "cell2"], ["cell1"], ["cell1", "cell2"]]

        Note:
            If a cell from the pred_query is not present in the table, the method treats it as if it's in an imaginary
            row indexed as -1. Therefore, if you see a [-1] in the result, it means one or more cells in your pred_query
            did not appear in the table.
        """
        # Initializing a defaultdict to store the results of the query
        query_ans = defaultdict(list)
        # Splitting the query into cells
        cells: list = pred_query.split(",")
        # Counting the occurrences of each cell in the query
        counted_cells = Counter(cells)
        # Iterating over each cell type and its count from the counted_cells
        for cell, count in counted_cells.items():
            # Finding the row ids where the current cell type exists in the table
            row_ids = np.where(table == cell.strip())[0]
            if len(row_ids) == 0:
                # If the cell is not present in the table, set the row_id as -1
                row_ids = [-1]

            # If the count of cell in the query > 1, select the first 'count' number of rows
            if count > 1:
                row_ids = row_ids[:count]
            # Appending the cell to the rows in the query_ans for each row_id
            [query_ans[idx].append(cell.strip()) for idx in row_ids]
        # Return the aggregated cells from each row as a list of lists
        return list(query_ans.values())
