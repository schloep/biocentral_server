import torch
import numpy as np

from typing import List, Dict
from biotrainer.protocols import Protocol
from biotrainer.utilities import get_device

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import load_onnx_model, to_cpu, get_batched_data


class ProtT5Conservation(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.model = load_onnx_model(model_name=self.get_metadata().name)
        self.device = get_device()
        self.non_padded_embedding_lengths = {}  # Undo padding after predictions

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ProtT5Conservation",
            protocol=Protocol.residue_to_class,
            description='',
            authors='C{\'{e}}line Marquet and Michael Heinzinger and Tobias Olenyi and Christian Dallago and Kyra Erckert and Michael Bernhofer and Dmitrii Nechaev and Burkhard Rost',
            model_link='https://github.com/Rostlab/VESPA',
            citation='https://doi.org/10.1007/s00439-021-02411-y',
            licence='Apache License',
            description_return_values='',
            model_size='',
            testset_performance='',
            training_data_link='http://data.bioembeddings.com/public/design/',
            embedder='Rostlab/prot_t5_xl_uniref50'
        )

    def _prepare_inputs(self, embeddings):
        self.non_padded_embedding_lengths = {idx: embedding.shape[0] for idx, embedding in embeddings.items()}
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=False)

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        results = []
        for batch in inputs:
            cons_Yhat = self.model.run(None, batch)
            cons_Yhat = torch.from_numpy(np.float32(np.stack(cons_Yhat[0])))
            cons_Yhat = to_cpu(torch.max(cons_Yhat, dim=-1, keepdim=True)[1]).astype(np.byte)
            results.extend(list(cons_Yhat))
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids: List[str]):
        formatted_predictions = {}
        for embd_idx, pred in enumerate(model_output):
            embedding_id = embedding_ids[embd_idx]
            formatted_predictions[embedding_id] = [''.join([str(yhat) for pred_idx, yhat in enumerate(pred) if
                                                            pred_idx < self.non_padded_embedding_lengths[
                                                                embedding_id]])]
        return formatted_predictions
