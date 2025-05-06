from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def _prepare_inputs(self, embeddings):
        pass

    @abstractmethod
    def predict(self, embeddings):
        pass

    @abstractmethod
    def _post_process(self, model_output, embedding_ids):
        pass
