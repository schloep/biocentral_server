import torch
import numpy as np
from scipy.special import softmax

from src.biotrainer.biotrainer.utilities import get_device
from ..utils import load_multiple_onnx_models, to_cpu, AvailableModels, get_batched_data
from base_model import BaseModel
from tmbed_viterbi import Decoder

class TMbed(BaseModel):
    def __init__(self, batch_size):
        # TODO: ist das schön mit der batch_size durch super? Pro: wird erzwungen, con: unübersichtlich
        super().__init__(batch_size=batch_size)
        self.models = load_multiple_onnx_models(model_name=AvailableModels.TMbed.value())
        self.decoder = Decoder()
        self.device = get_device()
        self.pred2label = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'S', 5: 'i', 6: 'o'}

    def _prepare_inputs(self, embeddings):
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=True)

    def predict(self, embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = embeddings.keys()
        results = []
        for batch in inputs:
            B, L, _ = batch['input'].shape
            ensemble_container = torch.zeros((B, 5, L), device=self.device, dtype=torch.float32)
            for model in self.models:
                y = model.run(None, batch)
                # TODO: hier 'from_numpy' und in 'to_cpu' wieder zu numpy? Funktioniert das auch ohne?
                # y = torch.from_numpy(np.float32(np.stack(y[0])))
                ensemble_container = ensemble_container + softmax(np.stack(y[0]), axis=1)
            probabilities = (ensemble_container / len(self.models))
            mem_Yhat = to_cpu(self.decoder(probabilities, batch['mask'])).astype(np.byte)
            # TODO: test if shape is B x residue_preds (also [[1,0,0,...], [...]])
            results.extend(mem_Yhat)  # -> no batches
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids):
        formatted_predictions = {}
        for i, pred in enumerate(model_output):
            formatted_predictions[embedding_ids[i]] = ''.join([self.pred2label[j] for j in pred])
        return formatted_predictions
