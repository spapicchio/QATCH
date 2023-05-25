import random
from abc import ABC, abstractmethod

from ..database_reader import SingleDatabase


class AbstractSqlGenerator(ABC):
    def __init__(self, database: SingleDatabase, seed=2023):
        self.database = database
        random.seed(seed)

    @abstractmethod
    def sql_generate(self, table_name: str) -> tuple[list, list, list, list]:
        raise NotImplementedError
