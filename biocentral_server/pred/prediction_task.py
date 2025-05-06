from typing import Callable, Dict, Optional
from biotrainer.utilities import get_device

from ..server_management import TaskInterface, TaskDTO
from ..embeddings import LoadEmbeddingsTask
from utils import get_model, AvailableModels
from models.base_model import BaseModel
from biotrainer.protocols import Protocol
from metadata_endpoint import ModelMetadata


class PredictionTask(TaskInterface):
    def __init__(self, model_data: dict[AvailableModels, ModelMetadata], sequence_input, reduced):
        self.model_data = model_data
        self.sequence_input = sequence_input
        self.reduced = reduced
        self.device = get_device()

    def run_task(self, update_dto_callback: Callable) -> TaskDTO:
        # wenn reduced true dann sind das per sequence embeddings, sonst per residue
        reduced = True if all(
            [model.protocol in Protocol.using_per_sequence_embeddings() for model in
             self.model_data.values()]) else False
        embeddings = self._pre_embed_with_db(reduced=reduced)
        # Per-Sequence: Batch x Embedding_Dim
        # Per-Residue: Batch x Seq_Length x Embedding_Dim
        # Alle Models die per sequence sind im preprocessing embeddins.mean machen um das zu reducen aber nur wenn die Shape der Embeddings wie von per-residue ist

        predictions = {}
        for model_name, model_metadata in self.model_data.items():
            model = get_model(model_name=model_name.value())
            predictions[model_name.value()] = model.predict(embeddings=embeddings)
        return TaskDTO.finished(result={"predictions": predictions})

    def _pre_embed_with_db(self, reduced):
        # TODO: add correct embedder name -> ProtT5 or ESM-2
        load_embeddings_task = LoadEmbeddingsTask(embedder_name=self.model.embedder_name,
                                                  sequence_input=self.sequence_input,
                                                  reduced=reduced,
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
