import torch
import numpy as np

from biotrainer.utilities import get_device
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import load_onnx_model, to_cpu, get_batched_data


class SETH(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.model = load_onnx_model(model_name=self.get_metadata().name)
        self.device = get_device()

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

    def _prepare_inputs(self, embeddings):
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=False)

    def predict(self, embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        results = []
        for batch in inputs:
            diso_Yhat = self.model.run(None, batch)
            diso_Yhat = to_cpu(torch.from_numpy(np.float32(np.stack(diso_Yhat[0]))))
            results.extend(list(diso_Yhat))
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids):
        formatted_predictions = {}
        for i, pred in enumerate(model_output):
            formatted_predictions[embedding_ids[i]] = [', '.join([str(z_score) for z_score in pred])]
        print(formatted_predictions)  # TODO
        return formatted_predictions
