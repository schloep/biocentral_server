from typing import Callable

from biotrainer.utilities import get_device
from .metadata_endpoint import ModelMetadata
from .single_prediction_task import SinglePredictionTask
from .models import AvailableModels, get_model

from ..server_management import TaskInterface, TaskDTO


class MultiPredictionTask(TaskInterface):
    def __init__(self, model_data: dict[str, ModelMetadata], sequence_input, batch_size):
        self.model_data = model_data
        self.sequence_input = sequence_input
        self.device = get_device()
        self.batch_size = batch_size

    def run_task(self, update_dto_callback: Callable) -> TaskDTO:
        predictions = {}
        for model_name, model_metadata in self.model_data.items():
            model = get_model(model_name=model_name, batch_size=self.batch_size)
            single_pred_task = SinglePredictionTask(model=model,
                                                    embedder_name=model_metadata.embedder,
                                                    sequence_input=self.sequence_input,
                                                    model_protocol=model_metadata.protocol,
                                                    device=self.device)
            load_dto = None
            for dto in self.run_subtask(single_pred_task):
                load_dto = dto
            if not load_dto:
                return TaskDTO.failed(error=f"Model prediction with the {model_name} model failed.")
            single_prediction = load_dto.update["prediction"]
            predictions[model_name] = single_prediction

        return TaskDTO.finished(result={"predictions": predictions})
