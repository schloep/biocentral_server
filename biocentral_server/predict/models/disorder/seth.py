import torch
import numpy as np

from typing import List, Dict
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import to_cpu


class SETH(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, uses_ensemble=False, requires_mask=False, requires_transpose=False)

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

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        results = []
        for batch in inputs:
            diso_Yhat = self.model.run(None, batch)
            diso_Yhat = to_cpu(torch.from_numpy(np.float32(np.stack(diso_Yhat[0]))))
            results.extend(list(diso_Yhat))
        model_output = {"disorder": results}
        return self._post_process(model_output=model_output, embedding_ids=embedding_ids, delimiter=",")
