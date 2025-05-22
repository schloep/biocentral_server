import torch
import numpy as np

from typing import Dict
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata


class LightAttentionSubcellularLocalization(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, uses_ensemble=False,
                         requires_mask=True, requires_transpose=True, model_dir_name="la_subcell")
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

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="LightAttentionSubcellularLocalization",
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

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        results = []
        for batch in inputs:
            batch = self._transpose_batch(batch)

            subcell_Yhat = self.model.run(None, batch)
            subcell_Yhat = torch.from_numpy(np.float32(np.stack(subcell_Yhat[0])))
            subcell_Yhat = self._finalize_raw_prediction(torch.max(subcell_Yhat, dim=1)[1], dtype=np.byte)

            results.extend(subcell_Yhat)

        model_output = {"subcellular_localization": results}
        return self._post_process(model_output=model_output, embedding_ids=embedding_ids,
                                  label_maps={"subcellular_localization": self.class2label_subcell})
