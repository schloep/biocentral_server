import torch
import numpy as np

from ..utils import to_cpu, get_batched_data, load_onnx_model
from .base_model import BaseModel


class LightAttention(BaseModel):
    name = 'LightAttention'

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.la_subcell = load_onnx_model(model_name='la_subcell')
        self.la_mem = load_onnx_model(model_name='la')
        self.device = ""
        self.embedder_name = "Rostlab/prot_t5_xl_uniref50"  # TODO Huggingface name
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

    def _prepare_inputs(self, embeddings):
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=True)

    def predict(self, embeddings):
        # TODO: get rid of to_numpy, from numpy...
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = embeddings.keys()
        results = []
        for batch in inputs:
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

    def _post_process(self, model_output, embedding_ids):
        formatted_predictions = {}
        for i, pred in enumerate(model_output):
            formatted_predictions[embedding_ids[i]] = {
                'subcell': self.class2label_subcell[pred['subcell']],
                'mem': self.class2label_mem[pred['mem']]
            }
        return formatted_predictions
