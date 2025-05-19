from typing import List, Dict
from abc import ABC, abstractmethod

from .model_metadata import ModelMetadata


class BaseModel(ABC):

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @staticmethod
    @abstractmethod
    def get_metadata() -> ModelMetadata:
        raise NotImplementedError

    @abstractmethod
    def _prepare_inputs(self, embeddings):
        raise NotImplementedError

    @abstractmethod
    def predict(self, sequences: Dict[str, str], embeddings):
        raise NotImplementedError

    @abstractmethod
    def _post_process(self, model_output, embedding_ids: List[str]):
        raise NotImplementedError
