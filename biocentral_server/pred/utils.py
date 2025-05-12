from enum import StrEnum
from typing import Any

import numpy as np

from models.tmbed import TMbed
from models.light_attention import LightAttention
from models.seth import SETH
from models.bind_predict import BindEmbeDL
from models.conservation import Conservation
from pathlib import Path
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile


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
    AvailableModels.BindEmbeDL: BindEmbeDL,
    AvailableModels.Conservation: Conservation,
}
MODEL_PATH = "assets/models"


def get_model(model_name: str, batch_size):
    model_class = MODEL_REGISTRY[AvailableModels(model_name)]
    if not model_class:
        raise ValueError(f'Model {model_name} not found in registry.')

    return model_class(batch_size=batch_size)


def load_multiple_onnx_models(model_name):
    models = []
    model_dir = f"{MODEL_PATH}/{model_name.str.lower()}"
    for onnx_file in Path(model_dir).iterdir():
        try:
            onnx_model = ort.InferenceSession(onnx_file)
            models.append(onnx_model)
        except NoSuchFile:
            print(f'ERROR: No onnx model at path {onnx_file}.')
            quit()
    return models


def load_onnx_model(model_name):
    model_dir = f"{MODEL_PATH}/{model_name.str.lower()}"
    return ort.InferenceSession(model_dir)


def to_cpu(tensor):
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()


def get_batched_data(batch_size: int, data: np.array, mask: bool = False):
    batched_data = []
    if mask:
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            padded_embeddings_batch, attention_masks_batch = pad_embeddings(embeddings=batch_data,
                                                                            get_attention_mask=True)
            batched_data.append(
                {
                    'input': padded_embeddings_batch,
                    'mask': attention_masks_batch
                })
    else:
        for i in range(0, len(data), batch_size):
            batched_data.append(
                {
                    'input': pad_embeddings(embeddings=data[i:i + batch_size]),
                })
    return batched_data


def pad_embeddings(embeddings: np.array, get_attention_mask: bool = False):
    # TODO: better function naming
    max_length = max(array.shape[0] for array in embeddings)
    padded_arrays = []
    attention_masks = []
    for array in embeddings:
        # pad embeddings
        padding = ((0, max_length - array.shape[0]), (0, 0))  # ((Pad oben, Pad unten), (Pad links, Pad rechts))
        padded_array = np.pad(array, padding, mode='constant', constant_values=0)
        padded_arrays.append(padded_array)

        if get_attention_mask:
            # create attention masks
            attention_mask = np.ones(array.shape[0], dtype=int)
            pad_mask = np.zeros(max_length - array.shape[0], dtype=int)
            full_attention_mask = np.concatenate([attention_mask, pad_mask])
            attention_masks.append(full_attention_mask)
    padded_embeddings = np.stack(padded_arrays)
    if get_attention_mask:
        attention_masks_numpy = np.float32(np.stack(attention_masks))
        return padded_embeddings, attention_masks_numpy
    else:
        return padded_embeddings
