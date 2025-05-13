from enum import StrEnum
from typing import Any

from .tmbed import TMbed
from .light_attention import LightAttention
from .seth import SETH
from .bind_predict import BindEmbed
from .conservation import Conservation

class AvailableModels(StrEnum):
    TMbed = 'TMbed'
    LightAttention = 'LightAttention'
    Conservation = 'Conservation'
    SecondaryStructure = 'SecondaryStructure'
    BindEmbeDL = 'BindEmbeDL'
    SETH = 'SETH'
    VespaG = 'VespaG'


MODEL_REGISTRY: dict[AvailableModels, Any] = {
    AvailableModels.TMbed: TMbed,
    AvailableModels.LightAttention: LightAttention,
    AvailableModels.SETH: SETH,
    AvailableModels.BindEmbeDL: BindEmbed,
    AvailableModels.Conservation: Conservation,
}


def get_model(model_name: str, batch_size):
    model_class = MODEL_REGISTRY[AvailableModels(model_name)]
    if not model_class:
        raise ValueError(f'Model {model_name} not found in registry.')

    return model_class(batch_size=batch_size)