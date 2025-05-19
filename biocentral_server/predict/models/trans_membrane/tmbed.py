import torch
import numpy as np

from tmbed import Decoder
from typing import List, Dict
from scipy.special import softmax
from biotrainer.protocols import Protocol
from biotrainer.utilities import get_device

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import load_multiple_onnx_models, to_cpu, get_batched_data


class TMbed(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.models = load_multiple_onnx_models(model_name=self.get_metadata().name)
        self.decoder = Decoder()
        self.device = get_device()
        self.pred2label = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'S', 5: 'i', 6: 'o'}
        self.non_padded_embedding_lengths = {}  # Undo padding after predictions

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="TMbed",
            protocol=Protocol.residue_to_class,
            description='',
            authors='Bernhofer, Michael and Rost, Burkhard',
            model_link='https://github.com/BernhoferM/TMbed',
            citation='https://doi.org/10.1101/2022.06.12.495804',
            licence='Apache License',
            description_return_values='',
            model_size='',
            testset_performance='',
            training_data_link='http://data.bioembeddings.com/public/design/',
            embedder='Rostlab/prot_t5_xl_uniref50'
        )

    def _prepare_inputs(self, embeddings):
        self.non_padded_embedding_lengths = {idx: embedding.shape[0] for idx, embedding in embeddings.items()}
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=True)

    @staticmethod
    def _transpose_batch(batch):
        return {k: v.transpose(0, 2, 1) if k == "input" else v for k, v in batch.items()}

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = embeddings.keys()
        results = []
        for batch in inputs:
            B, L, _ = batch['input'].shape

            # Container for summing up predictions of individual models in the ensemble
            ensemble_container = torch.zeros((B, len(self.pred2label.keys()), L), device=self.device,
                                             dtype=torch.float32)
            for model in self.models:
                y = model.run(None, batch)
                # TODO [Refactoring] Avoid unnecessary casting from numpy to pytorch and vice versa
                y = torch.from_numpy(np.float32(np.stack(y[0])))
                ensemble_container = ensemble_container + softmax(np.stack(y[0]), axis=1)
            probabilities = (ensemble_container / len(self.models))
            mem_Yhat = to_cpu(self.decoder(probabilities, batch['mask'])).astype(np.byte)
            results.extend(list(mem_Yhat))  # -> no batches
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids: List[str]):
        formatted_predictions = {}
        for embed_idx, pred in enumerate(model_output):
            embedding_id = embedding_ids[embed_idx]
            formatted_predictions[embedding_id] = ''.join([self.pred2label[j] for pred_idx, j in enumerate(pred) if
                                                           pred_idx < self.non_padded_embedding_lengths[embedding_id]])
        return formatted_predictions
