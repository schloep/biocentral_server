from typing import Union
import numpy as np
from pathlib import Path
import onnxruntime as ort
from jedi.inference.gradual.typing import Tuple
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

MODEL_PATH = "assets/models"


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
    model_dir = f"{MODEL_PATH}/{model_name.lower()}/{model_name.lower()}.onnx"  # TODO
    return ort.InferenceSession(model_dir)


def to_cpu(tensor):
    # This function is copied from the PGP-repo to recreate the exact same inference as tested
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()


def get_batched_data(batch_size: int, data: np.array, mask: bool = False) -> list[dict]:
    """
    Returns the given data in batches. Each batch contains its data as a dict. The structure is enforced by the onnx runtime model.
    :param batch_size: The number of elements per batch
    :param data: The already embedded data
    :param mask: True if the onnx-model requires a mask, else false.
    :return: A list of dicts containing the batched data:
        'input': <batch_of_embeddings>
        and additionally if mask required:
        'mask': <attention_mask>
        The keys must be named like that, or else the onnx-model-inference will fail.
    """
    batched_data = []
    data = list(data)
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
                    'input': pad_embeddings(embeddings=data[i:i + batch_size], get_attention_mask=False)[0],
                })
    return batched_data


def pad_embeddings(embeddings: np.array, get_attention_mask: bool = False):
    """
    Padds the given batch of embeddings to the longest given sequence. Creates the corresponding attention mask if needed.
    :param embeddings: Batch of embeddings to pad
    :param get_attention_mask: True if the attention mask is needed, else false
    :return: A tuple containing the padded batch of embeddings and the corresponding attention mask if get_attention_mask == True, else
    the padded embeddings and an empty list.
    """
    max_length = max(array.shape[0] for array in embeddings)
    padded_arrays = []
    attention_masks = []
    for array in embeddings:
        padding = ((0, max_length - array.shape[0]), (0, 0))
        padded_array = np.pad(array, padding, mode='constant', constant_values=0)
        padded_arrays.append(padded_array)

        if get_attention_mask:
            attention_mask = np.ones(array.shape[0], dtype=int)
            pad_mask = np.zeros(max_length - array.shape[0], dtype=int)
            full_attention_mask = np.concatenate([attention_mask, pad_mask])
            attention_masks.append(full_attention_mask)
    padded_embeddings = np.stack(padded_arrays)
    if get_attention_mask:
        attention_masks = np.float32(np.stack(attention_masks))
    return padded_embeddings, attention_masks
