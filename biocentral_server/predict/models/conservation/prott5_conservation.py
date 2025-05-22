import torch
import numpy as np

from typing import List, Dict
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata, Prediction


class ProtT5Conservation(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, uses_ensemble=False, requires_mask=False, requires_transpose=False)

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

    def predict(self, sequences: Dict[str, str], embeddings) -> Dict[str, List[Prediction]]:
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        results = []
        for batch in inputs:
            cons_Yhat = self.model.run(None, batch)
            cons_Yhat = torch.from_numpy(np.float32(np.stack(cons_Yhat[0])))
            cons_Yhat = self._finalize_raw_prediction(torch.max(cons_Yhat, dim=-1, keepdim=True)[1], dtype=np.byte)
            results.extend(cons_Yhat)
        model_output = {"conservation": results}
        return self._post_process(model_output=model_output, embedding_ids=embedding_ids)
