import torch
import numpy as np

from biotrainer.utilities import get_device
from ..utils import load_multiple_onnx_models, to_cpu, get_batched_data
from .base_model import BaseModel
from torch import nn


class BindEmbed(BaseModel):
    name = "BindEmbed"

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.models = load_multiple_onnx_models(model_name=self.name)
        self.device = get_device()
        self.sigmoid = nn.Sigmoid()
        self.binding_classes = {0: ('metal', "M"), 1: ('nucleic', "N"), 2: ('small', "S")}

    def _prepare_inputs(self, embeddings):
        embeddings_transposed = {embedding_id: torch.permute(embedding, (0, 2, 1)) for embedding_id, embedding in
                                 embeddings.items()}
        return get_batched_data(batch_size=self.batch_size, data=embeddings_transposed.values(), mask=False)

    def predict(self, embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = embeddings.keys()
        results = []
        for batch in inputs:
            B, L, _ = batch['input'].shape
            # container for adding predictions of individual models in the ensemble
            ensemble_container = torch.zeros((B, 3, L), device=self.device, dtype=torch.float16)
            for model in self.models:  # for each model in the ensemble
                model_output_numpy = model.run(None, batch)
                model_output_torch = torch.from_numpy(np.float32(np.stack(model_output_numpy[0])))
                pred = self.sigmoid(model_output_torch)
                pred = torch.from_numpy(np.float32(np.stack(pred)))
                ensemble_container = ensemble_container + pred
            # normalize
            bind_Yhat = ensemble_container / len(self.models)
            # B x 3 x L --> B x L x 3
            bind_Yhat = torch.permute(bind_Yhat, (0, 2, 1))
            bind_Yhat = to_cpu(bind_Yhat > 0.5).astype(np.byte)
            results.extend(list(bind_Yhat))
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids):
        formatted_predictions = {}
        for idx, (binding_type, bind_short) in self.binding_classes.items():
            for i, pred in enumerate(model_output):
                formatted_predictions[binding_type][embedding_ids[i]] = ''.join([bind_short if j==1 else "-" for j in pred[:,idx]])
        return formatted_predictions
