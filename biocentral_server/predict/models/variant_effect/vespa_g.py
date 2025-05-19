import torch
import numpy as np

from typing import List, Dict
from biotrainer.protocols import Protocol
from vespag import ScoreNormalizer, SAV, compute_mutation_score, mask_non_mutations

from ..base_model import BaseModel, ModelMetadata

from ...model_utils import load_onnx_model


class VespaG(BaseModel):

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.model = load_onnx_model(model_name=self.get_metadata().name)
        self.GEMME_ALPHABET = "ACDEFGHIKLMNPQRSTVWY" # TODO Should be imported from VespaG
        self.AMINO_ACIDS = sorted(self.GEMME_ALPHABET)
        self.zero_based_mutations = False
        self.mutations_per_protein = {}  # Calculated during predict
        self.normalizer = ScoreNormalizer('minmax')

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="VespaG",
            protocol=Protocol.residue_to_class,  # ?
            description='',
            authors='',
            model_link='https://iteragit.iteratec.de/biocentral-at-iteratec/vespag/-/blob/export_onnx/README.md?ref_type=heads',
            citation='https://doi.org/10.1093/bioinformatics/btae621',
            licence='GNU GENERAL PUBLIC LICENSE',
            description_return_values='',
            model_size='',
            testset_performance='',
            training_data_link='https://zenodo.org/records/11085958',
            embedder="facebook/esm2_t33_650M_UR50D"#'facebook/esm2_t36_3B_UR50D'
        )

    def _prepare_inputs(self, embeddings):
        #return [{'input': embedding.unsqueeze(0).numpy()} for embedding in embeddings.values()] TODO ESM LARGE
        return [{'input': torch.repeat_interleave(embedding, 2, dim=-1).unsqueeze(0).numpy()}
                for embedding in embeddings.values()]

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        self.mutations_per_protein = {  # TODO Should be handled in VespaG
            protein_id: [
                SAV(i, wildtype_aa, other_aa, not self.zero_based_mutations)
                for i, wildtype_aa in enumerate(sequence)
                for other_aa in self.AMINO_ACIDS
                if other_aa != wildtype_aa
            ]
            for protein_id, sequence in sequences.items()
        }
        vespag_scores = {}
        for seq_idx, sequence_embedding in enumerate(inputs):
            seq_id = embedding_ids[seq_idx]
            y = self.model.run(None, sequence_embedding)
            y = torch.from_numpy(np.float32(np.stack(y[0]))).squeeze(0)
            y = mask_non_mutations(y, sequences[seq_id])
            vespag_scores[seq_id] = y.detach().numpy()
        self.normalizer.fit(np.concatenate([y.flatten() for y in vespag_scores.values()]))
        return self._post_process(model_output=vespag_scores, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids: List[str]):
        scores_per_protein = {}
        transform_scores = True  # default value in the VespaG repo
        for seq_id, y in model_output.items():
            scores_per_protein[seq_id] = {
                mutation: compute_mutation_score(
                    y,
                    mutation,
                    transform=transform_scores,
                    normalizer=self.normalizer,
                )
                for mutation in self.mutations_per_protein[seq_id]
            }
        return scores_per_protein
