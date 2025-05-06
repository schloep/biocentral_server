import torch
import numpy as np

from ..utils import to_cpu
from base_model import BaseModel
from ..utils import load_onnx_model
from ..metadata_endpoint import get_metadata


class LightAttention(BaseModel):
    def __init__(self):
        self.la_subcell = load_onnx_model(model_name='la_subcell')
        self.la_mem = load_onnx_model(model_name='la')
        self.device = ""
        self.embedder_name = get_metadata()['LightAttention']['embedder']
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
        ort_inputs = {'input': embeddings.numpy(),
                      'mask': attention_mask.numpy()}
        return ort_inputs

    def predict(self, embeddings):
        # TODO: get rid of to_numpy, from numpy...
        inputs = self._prepare_inputs(embeddings=embeddings)
        subcell_Yhat = self.la_subcell.run(None, inputs)
        subcell_Yhat = torch.from_numpy(np.float32(np.stack(subcell_Yhat[0])))
        subcell_Yhat = to_cpu(torch.max(subcell_Yhat, dim=1)[1]).astype(np.byte)

        la_mem_Yhat = self.la_mem.run(None, inputs)
        la_mem_Yhat = torch.from_numpy(np.float32(np.stack(la_mem_Yhat[0])))
        la_mem_Yhat = to_cpu(torch.max(la_mem_Yhat, dim=1)[1]).astype(np.byte)

        return self._post_process(model_output={
            'subcell': subcell_Yhat,
            'mem': la_mem_Yhat
        })

    def _post_process(self, model_output):
        # mehrere Sequence Ids -> weil mehrere Sequences in embeddings
        return {sequence_id: {
            'subcell': '\n'.join([self.class2label_subcell[pred] for pred in model_output['subcell']]),
            'mem': '\n'.join([self.class2label_mem[pred] for pred in model_output['mem']])
        }
        }
