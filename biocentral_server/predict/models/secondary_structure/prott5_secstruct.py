import torch
import numpy as np

from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import load_onnx_model, get_batched_data, to_cpu


class ProtT5SecondaryStructure(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.model = load_onnx_model(model_name=self.get_metadata().name)
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

    def _prepare_inputs(self, embeddings):
        return get_batched_data(batch_size=self.batch_size, data=embeddings.values(), mask=False)

    def predict(self, embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = embeddings.keys()
        results = []
        for batch in inputs:
            d3_Yhat, d8_Yhat = self.model.run(None, batch)
            d3_Yhat = torch.from_numpy(np.float32(np.stack(d3_Yhat)))
            d8_Yhat = torch.from_numpy(np.float32(np.stack(d8_Yhat)))
            d3_Yhat = to_cpu(torch.max(d3_Yhat, dim=-1, keepdim=True)[1]).astype(np.byte)
            d8_Yhat = to_cpu(torch.max(d8_Yhat, dim=-1, keepdim=True)[1]).astype(np.byte)
            batch_result = [{'d3_Yhat': d3_Yhat_single, 'd8_Yhat': d8_Yhat_single} for d3_Yhat_single, d8_Yhat_single in
                            zip(list(d3_Yhat), list(d8_Yhat))]
            results.extend(batch_result)
        return self._post_process(model_output=results, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids):
        formatted_predictions = {}
        for i, pred in enumerate(model_output):
            formatted_predictions[embedding_ids[i]] = {
                'd3_Yhat': ''.join([self.label_mapping_3_states[j] for j in pred['d3_Yhat']]),
                'd8_Yhat': ''.join([self.label_mapping_8_states[j] for j in pred['d8_Yhat']])
            }
        return formatted_predictions
