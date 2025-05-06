from typing import Callable
from biotrainer.protocols import Protocol

from biocentral_server.embeddings import LoadEmbeddingsTask
from .models.base_model import BaseModel
from ..server_management import TaskInterface, TaskDTO


class SinglePredictionTask(TaskInterface):
    def __init__(self, model: BaseModel, embedder_name, sequence_input, model_protocol, device):
        self.model = model
        self.embedder_name = embedder_name
        self.sequence_input = sequence_input
        self.reduced = True if model_protocol in Protocol.using_per_sequence_embeddings() else False
        self.device = device

    def run_task(self, update_dto_callback: Callable) -> TaskDTO:
        embeddings = self._embed_sequences()
        predictions = self.model.predict(embeddings=embeddings)
        return TaskDTO.finished(result={"predictions": predictions})

    def _embed_sequences(self):
        load_embeddings_task = LoadEmbeddingsTask(embedder_name=self.embedder_name,
                                                  sequence_input=self.sequence_input,
                                                  reduced=self.reduced,
                                                  use_half_precision=False,
                                                  device=self.device)
        load_dto = None
        for dto in self.run_subtask(load_embeddings_task):
            load_dto = dto

        if not load_dto:
            return TaskDTO.failed(error="Loading of embeddings failed before export!")

        missing = load_dto.update["missing"]
        embeddings = load_dto.update["embeddings"]
        if len(missing) > 0:
            return TaskDTO.failed(error=f"Missing number of embeddings before export: {len(missing)}")

        return {triple.id: triple.embd for triple in embeddings}
