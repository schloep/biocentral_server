import torch
import numpy as np

from typing import List, Dict
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata


class ProtT5SecondaryStructure(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, uses_ensemble=False, requires_mask=False, requires_transpose=False)
        self.label_mapping_3_states = {0: "H", 1: "E", 2: "L"}
        self.label_mapping_8_states = {idx: state for idx, state in enumerate("GHIBESTC")}

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ProtT5SecondaryStructure",
            protocol=Protocol.residue_to_class,
            description='',
            authors='',
            model_link='https://github.com/agemagician/ProtTrans',
            citation='https://doi.org/10.1109/TPAMI.2021.3095381',
            licence='Apache License',
            description_return_values='',
            model_size='',
            testset_performance='',
            training_data_link='http://data.bioembeddings.com/public/design/',
            embedder='Rostlab/prot_t5_xl_uniref50'
        )

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        model_output = {'d3_Yhat': [], 'd8_Yhat': []}
        for batch in inputs:
            d3_Yhat, d8_Yhat = self.model.run(None, batch)
            d3_Yhat = torch.from_numpy(np.float32(np.stack(d3_Yhat)))
            d8_Yhat = torch.from_numpy(np.float32(np.stack(d8_Yhat)))
            d3_Yhat = self._finalize_raw_prediction(torch.max(d3_Yhat, dim=-1, keepdim=True)[1], dtype=np.byte)
            d8_Yhat = self._finalize_raw_prediction(torch.max(d8_Yhat, dim=-1, keepdim=True)[1], dtype=np.byte)
            model_output['d3_Yhat'].extend(d3_Yhat)
            model_output['d8_Yhat'].extend(d8_Yhat)
        return self._post_process(model_output=model_output, embedding_ids=embedding_ids,
                                  label_maps={'d3_Yhat': self.label_mapping_3_states,
                                              'd8_Yhat': self.label_mapping_8_states}
                                  )
