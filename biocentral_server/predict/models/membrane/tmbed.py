import torch
import numpy as np

from tmbed import Decoder
from typing import List, Dict
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata



class TMbed(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size, uses_ensemble=True, requires_mask=True, requires_transpose=True)
        self.decoder = Decoder().to(self.device)
        self.pred2label = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'S', 5: 'i', 6: 'o'}

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

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        results = []
        for batch in inputs:
            B, L, _ = batch['input'].shape

            # Container for summing up predictions of individual models in the ensemble
            pred = torch.zeros((B, len(self.models), L), device=self.device)
            for model in self.models:
                y = model.run(None, batch)
                y = torch.from_numpy(np.float32(np.stack(y[0])))
                pred = pred + torch.softmax(y, dim=1).to(self.device)

            probabilities = (pred / len(self.models))
            mem_Yhat = self._finalize_raw_prediction(self.decoder(probabilities,
                                                                  torch.from_numpy(batch['mask']).to(self.device)),
                                                     dtype=np.byte)
            results.extend(mem_Yhat) # -> no batches
        model_output = {"trans_membrane": results}
        return self._post_process(model_output=model_output, embedding_ids=embedding_ids,
                                  label_maps={"trans_membrane": self.pred2label})
