import torch
import numpy as np

from typing import List, Dict
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import to_cpu, get_batched_data, load_onnx_model


class LightAttention(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.la_subcell = load_onnx_model(model_name='la_subcell')
        self.la_mem = load_onnx_model(model_name='la_mem')
        self.class2label_subcell = {
            0: "Cell_membrane",
            1: "Cytoplasm",
            2: "Endoplasmatic_reticulum",
            3: "Golgi_apparatus",
            4: "Lysosome_or_Vacuole",
            5: "Mitochondrion",
            6: "Nucleus",
            7: "Peroxisome",
            8: "Plastid",
            9: "Extracellular"
        }
        self.class2label_mem = {
            0: "Membrane",
            1: "Soluble"
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="LightAttention",
            protocol=Protocol.residues_to_class,
            description='',
            authors='Stärk, Hannes and Dallago, Christian and Heinzinger, Michael and Rost, Burkhard',
            model_link='https://github.com/HannesStark/protein-localization',
            citation=' https://doi.org/10.1093/bioadv/vbab035',
            licence='Apache License',
            description_return_values='',
            model_size='',  # onnx in MB
            testset_performance='',
            training_data_link='http://data.bioembeddings.com/public/design/',
            embedder='Rostlab/prot_t5_xl_uniref50'
        )

    def _prepare_inputs(self, embeddings):
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=True)

    @staticmethod
    def _transpose_batch(batch):
        return {k: v.transpose(0, 2, 1) if k == "input" else v for k, v in batch.items()}

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        results = []
        for batch in inputs:
            batch = self._transpose_batch(batch)

            subcell_Yhat = self.la_subcell.run(None, batch)
            subcell_Yhat = torch.from_numpy(np.float32(np.stack(subcell_Yhat[0])))
            subcell_Yhat = to_cpu(torch.max(subcell_Yhat, dim=1)[1]).astype(np.byte)

            la_mem_Yhat = self.la_mem.run(None, batch)
            la_mem_Yhat = torch.from_numpy(np.float32(np.stack(la_mem_Yhat[0])))
            la_mem_Yhat = to_cpu(torch.max(la_mem_Yhat, dim=1)[1]).astype(np.byte)
            batch_result = [{'subcell': subcell, 'mem': mem} for subcell, mem in
                            zip(list(subcell_Yhat), list(la_mem_Yhat))]
            results.extend(batch_result)
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids: List[str]):
        formatted_predictions = {}
        for i, pred in enumerate(model_output):
            formatted_predictions[embedding_ids[i]] = {
                'subcell': self.class2label_subcell[pred['subcell']],
                'mem': self.class2label_mem[pred['mem']]
            }
        return formatted_predictions
