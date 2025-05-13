from enum import StrEnum
from typing import Any

from .tmbed import TMbed
from .light_attention import LightAttention
from .seth import SETH
from .bind_embed import BindEmbed
from .conservation import Conservation
from .secondary_structure import SecondaryStructure
from .vespa_g import VespaG

class AvailableModels(StrEnum):
    TMbed = TMbed.name
    LightAttention = LightAttention.name
    Conservation = Conservation.name
    SecondaryStructure = SecondaryStructure.name
    BindEmbed = BindEmbed.name
    SETH = SETH.name
    VespaG = VespaG.name


MODEL_REGISTRY: dict[AvailableModels, Any] = {
    AvailableModels.TMbed: TMbed,
    AvailableModels.LightAttention: LightAttention,
    AvailableModels.SETH: SETH,
    AvailableModels.BindEmbed: BindEmbed,
    AvailableModels.Conservation: Conservation,
}


def get_model(model_name: str, batch_size):
    model_class = MODEL_REGISTRY[AvailableModels(model_name)]
    if not model_class:
        raise ValueError(f'Model {model_name} not found in registry.')

    return model_class(batch_size=batch_size)