import dataclasses

from flask import request, Blueprint, jsonify
from .metadata_endpoint import get_metadata
from .multi_prediction_task import MultiPredictionTask
from ..server_management import TaskManager

prediction_service_route = Blueprint("predict_rout", __name__)  # TODO

# Endpoint for ProtSpace dimensionality reduction methods for sequences
@prediction_service_route.route('/prediction_service/predict', methods=['POST'])
def predict():
    request_data = PredictionRequestData(**request.get_json())
    model_names = request_data.model_names
    if any(model_name not in get_metadata().keys() for model_name in model_names):
        return jsonify({"error": "Model not found"})
    else:
        model_data = {model_name: get_metadata()[model_name] for model_name in model_names}
        prediction_task = MultiPredictionTask(model_data=model_data,
                                              sequence_input=request_data.sequence_input,
                                              batch_size=request_data.batch_size)
        task_id = TaskManager().add_task(prediction_task)
        return jsonify({"task_id": task_id})


@dataclasses.dataclass
class PredictionRequestData:
    model_names: list[str]
    sequence_input: dict[str, str]  # sequence_id: sequence
    batch_size: int
