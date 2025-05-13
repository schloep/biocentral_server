from .metadata_endpoint import prediction_metadata_route
from .predict_endpoint import prediction_service_route

__all__ = [
    'prediction_metadata_route',
    'prediction_service_route'
]