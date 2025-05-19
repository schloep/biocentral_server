import torch
import numpy as np

from torch import nn
from typing import List, Dict
from biotrainer.utilities import get_device
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import load_multiple_onnx_models, to_cpu, get_batched_data


class BindEmbed(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.models = load_multiple_onnx_models(model_name=self.get_metadata().name)
        self.device = get_device()
        self.sigmoid = nn.Sigmoid()
        self.binding_classes = {0: ('metal', "M"), 1: ('nucleic', "N"), 2: ('small', "S")}
        self.non_padded_embedding_lengths = {}  # Undo padding after predictions

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="BindEmbed",
            protocol=Protocol.residue_to_class,
            description='',
            authors='Littmann, Maria and Heinzinger, Michael and Dallago, Christian and Weissenow, Konstantin and Rost, Burkhard',
            model_link='https://github.com/Rostlab/bindPredict/tree/e9f1f33c5b614966fbf7d85b79f856b68ca495ad',
            citation='https://doi.org/10.1038/s41598-021-03431-4',
            licence='Apache License',
            description_return_values='',
            model_size='',
            testset_performance='',
            training_data_link='http://data.bioembeddings.com/public/design/',
            embedder='Rostlab/prot_t5_xl_uniref50'
        )

    def _prepare_inputs(self, embeddings: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        self.non_padded_embedding_lengths = {idx: embedding.shape[0] for idx, embedding in embeddings.items()}
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=False)

    @staticmethod
    def _transpose_batch(batch):
        return {k: v.transpose(0, 2, 1) if k == "input" else v for k, v in batch.items()}

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        results = []
        for batch in inputs:
            B, L, _ = batch['input'].shape
            batch = self._transpose_batch(batch)

            # Container for summing up predictions of individual models in the ensemble
            ensemble_container = torch.zeros((B, len(self.binding_classes.keys()), L), device="cpu", dtype=torch.float16)
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

    def _post_process(self, model_output, embedding_ids: List[str]):
        formatted_predictions = {idx: {} for idx in embedding_ids}

        for binding_id, (binding_type, bind_short) in self.binding_classes.items():
            for i, pred in enumerate(model_output):
                embedding_id = embedding_ids[i]
                if binding_type not in formatted_predictions[embedding_id]:
                    formatted_predictions[embedding_id][binding_type] = ""

                formatted_predictions[embedding_id][binding_type] = (
                    ''.join([bind_short if j == 1 else "-" for pred_idx, j in enumerate(pred[:, binding_id])
                             if pred_idx < self.non_padded_embedding_lengths[embedding_id]]))
        return formatted_predictions
