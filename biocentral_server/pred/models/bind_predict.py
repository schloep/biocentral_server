import torch
import numpy as np
from scipy.special import softmax

from src.biotrainer.biotrainer.utilities import get_device
from ..utils import load_onnx_model, to_cpu, AvailableModels, get_batched_data
from base_model import BaseModel


class BindEmbeDL(BaseModel):
    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.model = load_onnx_model(model_name=AvailableModels.SETH.value())
        self.device = get_device()

    def _prepare_inputs(self, embeddings):
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=False)

    def predict(self, embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = embeddings.keys()
        results = []
        for batch in inputs:
            diso_Yhat = self.model.run(None, batch)
            diso_Yhat = to_cpu(torch.from_numpy(np.float32(np.stack(diso_Yhat[0]))))
            # TODO: test if this should be as in the pgp repo
            results.extend(diso_Yhat)
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids):
        formatted_predictions = {}
        for i, pred in enumerate(model_output):
            formatted_pred = [', '.join([str(j) for j in Zscore]) for Zscore in pred]
            formatted_predictions[embedding_ids[i]]: formatted_pred
        return formatted_predictions