import torch
import numpy as np

from typing import List, Dict
from biotrainer.protocols import Protocol
from biotrainer.utilities import get_device

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import load_onnx_model, to_cpu, get_batched_data


class SETH(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.model = load_onnx_model(model_name=self.get_metadata().name)
        self.device = get_device()
        self.non_padded_embedding_lengths = {}  # Undo padding after predictions

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SETH",
            protocol=Protocol.residue_to_class,
            description='',
            authors='Stärk, Hannes and Dallago, Christian and Heinzinger, Michael and Rost, Burkhard',
            model_link='https://github.com/DagmarIlz/SETH',
            citation='https://doi.org/10.1101/2022.06.23.497276 ',
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
            diso_Yhat = self.model.run(None, batch)
            diso_Yhat = to_cpu(torch.from_numpy(np.float32(np.stack(diso_Yhat[0]))))
            results.extend(list(diso_Yhat))
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids: List[str]):
        formatted_predictions = {}
        for embed_idx, pred in enumerate(model_output):
            embedding_id = embedding_ids[embed_idx]
            formatted_predictions[embedding_id] = [', '.join([str(z_score) for pred_idx, z_score in enumerate(pred) if
                                                              pred_idx < self.non_padded_embedding_lengths[
                                                                  embedding_id]])]
        return formatted_predictions
